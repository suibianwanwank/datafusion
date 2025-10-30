// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::{ArrayRef, ListArray};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datafusion_common::DFSchema;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_expr::{col, lit};
use datafusion_functions_nested::expr_fn::{array_has, array_has_all, array_has_any};
use datafusion_physical_expr::create_physical_expr;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

const MARKER: i32 = 42;

#[derive(Clone)]
struct MembershipData {
    batch: Arc<RecordBatch>,
    schema: Arc<Schema>,
}

fn row_value(row: usize, col: usize, row_len: usize) -> i32 {
    ((row * row_len + col) % 1024) as i32
}

fn build_rows(rows: usize, row_len: usize, null_ratio: f64, seed: u64) -> MembershipData {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut haystack_rows = Vec::with_capacity(rows);
    let mut any_rows = Vec::with_capacity(rows);
    let mut all_rows = Vec::with_capacity(rows);
    let mut other_rows = Vec::with_capacity(rows);

    for row in 0..rows {
        let is_null = rng.random_bool(null_ratio);
        if is_null {
            haystack_rows.push(None);
            any_rows.push(None);
            all_rows.push(None);
            other_rows.push(None);
            continue;
        }

        let mut values: Vec<Option<i32>> = (0..row_len)
            .map(|col| Some(row_value(row, col, row_len)))
            .collect();

        if let Some(last) = values.last_mut() {
            *last = Some(MARKER);
        }

        let hit_value = values
            .get(row_len.saturating_sub(1) / 2)
            .and_then(|v| *v)
            .unwrap_or(MARKER);
        let first_value = values.get(0).and_then(|v| *v).unwrap_or(MARKER);

        let mut other = values.clone();
        if let Some(first) = other.first_mut() {
            *first = Some(MARKER);
        }

        haystack_rows.push(Some(values.clone()));
        any_rows.push(Some(vec![Some(hit_value), Some(MARKER + 1)]));
        all_rows.push(Some(vec![Some(MARKER), Some(first_value)]));
        other_rows.push(Some(other));
    }

    let haystack = ListArray::from_iter_primitive::<Int32Type, _, _>(haystack_rows);
    let any_needles = ListArray::from_iter_primitive::<Int32Type, _, _>(any_rows);
    let all_needles = ListArray::from_iter_primitive::<Int32Type, _, _>(all_rows);
    let other = ListArray::from_iter_primitive::<Int32Type, _, _>(other_rows);

    let item_field = Arc::new(Field::new("item", DataType::Int32, true));
    let list_type = DataType::List(item_field);

    let schema = Arc::new(Schema::new(vec![
        Field::new("haystack", list_type.clone(), true),
        Field::new("needle_any", list_type.clone(), true),
        Field::new("needle_all", list_type.clone(), true),
        Field::new("other", list_type, true),
    ]));

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(haystack) as ArrayRef,
            Arc::new(any_needles) as ArrayRef,
            Arc::new(all_needles) as ArrayRef,
            Arc::new(other) as ArrayRef,
        ],
    )
    .expect("valid record batch");

    MembershipData {
        batch: Arc::new(batch),
        schema,
    }
}

fn build_physical_expr(
    expr: datafusion_expr::Expr,
    schema: &Arc<Schema>,
) -> Arc<dyn datafusion_physical_expr::PhysicalExpr> {
    let df_schema = Arc::new(
        DFSchema::try_from((**schema).clone()).expect("convert schema to DFSchema"),
    );
    let mut exec_props = ExecutionProps::new();
    exec_props.mark_start_execution(Arc::new(
        datafusion_common::config::ConfigOptions::default(),
    ));

    create_physical_expr(&expr, &df_schema, &exec_props).expect("create physical expr")
}

fn bench_expr(
    c: &mut Criterion,
    label: &str,
    expr: Arc<dyn datafusion_physical_expr::PhysicalExpr>,
    batch: Arc<RecordBatch>,
) {
    let expr = Arc::clone(&expr);
    let batch = Arc::clone(&batch);
    c.bench_function(label, move |b| {
        b.iter(|| {
            let value = expr
                .as_ref()
                .evaluate(batch.as_ref())
                .expect("expression evaluation");
            black_box(value);
        })
    });
}

fn bench_case(
    c: &mut Criterion,
    rows: usize,
    row_len: usize,
    null_ratio: f64,
    seed: u64,
) {
    let data = build_rows(rows, row_len, null_ratio, seed);

    let has_expr =
        build_physical_expr(array_has(col("haystack"), lit(MARKER)), &data.schema);
    let has_any_expr = build_physical_expr(
        array_has_any(col("haystack"), col("needle_any")),
        &data.schema,
    );
    let has_all_expr = build_physical_expr(
        array_has_all(col("haystack"), col("needle_all")),
        &data.schema,
    );

    let label_suffix = format!("rows={rows} len={row_len} null={:.1}", null_ratio);

    bench_expr(
        c,
        &format!("array_has_scalar {label_suffix}"),
        has_expr,
        Arc::clone(&data.batch),
    );
    bench_expr(
        c,
        &format!("array_has_any {label_suffix}"),
        has_any_expr,
        Arc::clone(&data.batch),
    );
    bench_expr(
        c,
        &format!("array_has_all {label_suffix}"),
        has_all_expr,
        Arc::clone(&data.batch),
    );
}

fn array_function_benchmark(c: &mut Criterion) {
    let mut seed = 7_u64;
    for &rows in &[512_usize, 4096] {
        for &row_len in &[8_usize, 32, 128] {
            for &null_ratio in &[0.0, 0.2] {
                bench_case(c, rows, row_len, null_ratio, seed);
                seed = seed.wrapping_add(1);
            }
        }
    }
}

criterion_group!(benches, array_function_benchmark);
criterion_main!(benches);
