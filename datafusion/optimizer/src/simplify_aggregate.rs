use crate::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion_common::{Column, Result};
use datafusion_expr::expr::{AggregateFunction, AggregateFunctionParams};
use datafusion_expr::{Aggregate, BinaryExpr, Expr, LogicalPlan, Operator, Projection};
use indexmap::IndexSet;
use std::sync::Arc;

#[derive(Default, Debug)]
pub struct SimplifyAggregate;

impl SimplifyAggregate {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for SimplifyAggregate {
    fn name(&self) -> &str {
        "simplify_aggregate"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        match &plan {
            LogicalPlan::Aggregate(aggregate) => {
                let mut projects = Vec::new();
                let mut aggregate_funcs = IndexSet::new();

                for group in &aggregate.group_expr {
                    let (q, n) = group.qualified_name();
                    projects.push(Expr::Column(Column::new(q, n)));
                }

                for agg_func in &aggregate.aggr_expr {
                    match agg_func {
                        Expr::AggregateFunction(a) if support_linear_optimize(a) => {
                            // Try to optimize the aggregate argument expression
                            if let Some(arg) = a.params.args.first() {
                                let mut temp_funcs = IndexSet::new();
                                if let Some(transformed_expr) =
                                    process_expr(arg, a, &mut temp_funcs)
                                {
                                    if !transformed_expr.column_refs().is_empty() {
                                        aggregate_funcs.extend(temp_funcs);
                                        projects.push(
                                            transformed_expr
                                                .alias(agg_func.name_for_alias()?),
                                        );
                                        continue;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    aggregate_funcs.insert(agg_func.clone());
                    let (q, n) = agg_func.qualified_name();
                    projects.push(Expr::Column(Column::new(q, n)));
                }

                if aggregate.aggr_expr.len() == aggregate_funcs.len()
                    && aggregate.aggr_expr.iter().eq(aggregate_funcs.iter())
                {
                    return Ok(Transformed::no(plan.clone()));
                }

                let new_aggregate = LogicalPlan::Aggregate(Aggregate::try_new(
                    Arc::clone(&aggregate.input),
                    aggregate.group_expr.clone(),
                    aggregate_funcs.into_iter().collect(),
                )?);

                let projection = LogicalPlan::Projection(Projection::try_new(
                    projects,
                    Arc::new(new_aggregate),
                )?);

                Ok(Transformed::yes(projection))
            }
            _ => Ok(Transformed::no(plan.clone())),
        }
    }
}

fn support_linear_optimize(agg_func: &AggregateFunction) -> bool {
    agg_func.func.is_linear()
        && agg_func.params.args.len() == 1
        && !agg_func.params.distinct
        && agg_func.params.filter.is_none()
        && agg_func.params.order_by.is_none()
        && agg_func.params.null_treatment.is_none()
}

fn process_expr(
    expr: &Expr,
    agg_func: &AggregateFunction,
    aggregates: &mut IndexSet<Expr>,
) -> Option<Expr> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
            Operator::Plus | Operator::Minus | Operator::Multiply | Operator::Divide => {
                let new_left = process_expr(left, agg_func, aggregates)?;
                let new_right = process_expr(right, agg_func, aggregates)?;
                Some(Expr::BinaryExpr(BinaryExpr {
                    left: Box::new(new_left),
                    op: *op,
                    right: Box::new(new_right),
                }))
            }
            _ => None,
        },
        Expr::Column(col) => {
            let base_agg = Expr::AggregateFunction(AggregateFunction {
                func: Arc::clone(&agg_func.func),
                params: AggregateFunctionParams {
                    args: vec![Expr::Column(col.clone())],
                    distinct: agg_func.params.distinct,
                    filter: agg_func.params.filter.clone(),
                    order_by: agg_func.params.order_by.clone(),
                    null_treatment: agg_func.params.null_treatment,
                },
            });
            aggregates.insert(base_agg.clone());
            base_agg
                .name_for_alias()
                .ok()
                .map(|alias| Expr::Column(Column::from_name(alias)))
        }
        Expr::Alias(alias) => Some(process_expr(&alias.expr, agg_func, aggregates)?),
        Expr::Literal(value) => Some(Expr::Literal(value.clone())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;
    use crate::test::*;
    use crate::OptimizerContext;
    use datafusion_common::Result;
    use datafusion_expr::test::function_stub::sum;
    use datafusion_expr::{col, lit, logical_plan::builder::LogicalPlanBuilder};
    use std::sync::Arc;

    fn observe(_plan: &LogicalPlan, _rule: &dyn OptimizerRule) {}

    fn assert_optimized_plan_eq(plan: LogicalPlan, expected: &str) -> Result<()> {
        let optimizer = Optimizer::with_rules(vec![Arc::new(SimplifyAggregate::new())]);
        let optimized_plan =
            optimizer.optimize(plan, &OptimizerContext::new(), observe)?;

        let formatted_plan = format!("{optimized_plan}");
        assert_eq!(formatted_plan, expected);
        Ok(())
    }

    #[test]
    fn simplify_multiple_linear_aggregates2() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![sum(col("b") * (lit(2)) + col("c") * (lit(3)))],
            )?
            .build()?;

        let expected = "Projection: test.a, sum(test.b) * Int32(2) + sum(test.c) * Int32(3) AS sum(test.b * Int32(2) + test.c * Int32(3))\
        \n  Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b), sum(test.c)]]\
        \n    TableScan: test";

        assert_optimized_plan_eq(plan, expected)
    }

    #[test]
    fn simplify_multiple_linear_aggregates3() -> Result<()> {
        let table_scan = test_table_scan()?;

        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![
                    sum(col("b") * (lit(2)) + col("c") * (lit(3))),
                    sum(col("b") * (lit(7)) - col("c") / (lit(3))),
                ],
            )?
            .build()?;

        let expected = "Projection: test.a, sum(test.b) * Int32(2) + sum(test.c) * Int32(3) AS sum(test.b * Int32(2) + test.c * Int32(3)), sum(test.b) * Int32(7) - sum(test.c) / Int32(3) AS sum(test.b * Int32(7) - test.c / Int32(3))\
        \n  Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b), sum(test.c)]]\
        \n    TableScan: test";

        assert_optimized_plan_eq(plan, expected)
    }
}
