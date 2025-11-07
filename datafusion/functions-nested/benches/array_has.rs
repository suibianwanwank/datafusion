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

use arrow::array::{Array, ArrayRef, Int32Builder, ListBuilder, StringBuilder};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datafusion_common::DFSchema;
use datafusion_expr::col;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_functions_nested::expr_fn::{array_has_all, array_has_any};
use datafusion_physical_expr::create_physical_expr;
use rand::distr::Alphanumeric;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

fn random_string(rng: &mut StdRng, len: usize) -> String {
    let value = rng.sample_iter(&Alphanumeric).take(len).collect();
    String::from_utf8(value).unwrap()
}

fn make_utf8_batch(row_count: usize, array_length: usize) -> RecordBatch {
    let mut rng = StdRng::seed_from_u64(120320);

    let mut haystack_builder = ListBuilder::new(StringBuilder::new());
    let mut needle_builder = ListBuilder::new(StringBuilder::new());

    let haystack_values: Vec<String> = (0..array_length)
        .map(|_| random_string(&mut rng, 8))
        .collect();

    (0..row_count).for_each(|_| {
        {
            let values_builder = haystack_builder.values();
            haystack_values
                .iter()
                .for_each(|value| values_builder.append_value(value));
        }
        haystack_builder.append(true);

        {
            let values_builder = needle_builder.values();
            (0..array_length / 2).for_each(|_| {
                let value = random_string(&mut rng, 5);
                values_builder.append_value(&value);
            });
        }
        needle_builder.append(true);
    });

    let needle_array = needle_builder.finish();
    let haystack_array = haystack_builder.finish();

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", needle_array.data_type().clone(), true),
        Field::new("b", haystack_array.data_type().clone(), true),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(needle_array) as ArrayRef,
            Arc::new(haystack_array) as ArrayRef,
        ],
    )
    .unwrap()
}

fn make_i32_batch(row_count: usize, array_length: usize) -> RecordBatch {
    let mut rng = StdRng::seed_from_u64(120320);

    let mut haystack_builder = ListBuilder::new(Int32Builder::new());
    let mut needle_builder = ListBuilder::new(Int32Builder::new());

    let haystack_values: Vec<i32> = (0..array_length)
        .map(|_| rng.random_range(0..200_000))
        .collect();

    (0..row_count).for_each(|_| {
        {
            let values_builder = haystack_builder.values();
            haystack_values
                .iter()
                .for_each(|value| values_builder.append_value(*value));
        }
        haystack_builder.append(true);

        {
            let values_builder = needle_builder.values();
            (0..array_length / 2).for_each(|_| {
                values_builder.append_value(rng.random_range(0..200_000));
            });
        }
        needle_builder.append(true);
    });

    let needle_array = needle_builder.finish();
    let haystack_array = haystack_builder.finish();

    let schema = Arc::new(Schema::new(vec![
        Field::new("a", needle_array.data_type().clone(), true),
        Field::new("b", haystack_array.data_type().clone(), true),
    ]));

    RecordBatch::try_new(
        schema,
        vec![Arc::new(needle_array), Arc::new(haystack_array)],
    )
    .unwrap()
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

/// Benchmarks array_has_all/any functions.
///
/// Run with:
/// ```sh
/// cargo bench --bench array_has
/// ```
fn criterion_benchmark(c: &mut Criterion) {
    let cases = &[(1024, 8), (1024, 64), (1024, 256)];
    for &(row_count, array_size) in cases {
        let utf8_batch = make_utf8_batch(row_count, array_size);
        let utf8_schema = utf8_batch.schema();
        {
            let expr =
                build_physical_expr(array_has_all(col("b"), col("a")), &utf8_schema);
            c.bench_function(
                &format!("array_has_all_utf8 {row_count}x{array_size}"),
                |b| b.iter(|| black_box(expr.evaluate(black_box(&utf8_batch)).unwrap())),
            );
        }
        {
            let expr =
                build_physical_expr(array_has_any(col("b"), col("a")), &utf8_schema);
            c.bench_function(
                &format!("array_has_any_utf8 {row_count}x{array_size}"),
                |b| b.iter(|| black_box(expr.evaluate(black_box(&utf8_batch)).unwrap())),
            );
        }

        let i32_batch = make_i32_batch(row_count, array_size);
        let i32_schema = i32_batch.schema();
        {
            let expr =
                build_physical_expr(array_has_all(col("b"), col("a")), &i32_schema);
            c.bench_function(
                &format!("array_has_all_int32 {row_count}x{array_size}"),
                |b| b.iter(|| black_box(expr.evaluate(black_box(&i32_batch)).unwrap())),
            );
        }
        {
            let expr =
                build_physical_expr(array_has_any(col("b"), col("a")), &i32_schema);
            c.bench_function(
                &format!("array_has_any_int32 {row_count}x{array_size}"),
                |b| b.iter(|| black_box(expr.evaluate(black_box(&i32_batch)).unwrap())),
            );
        }
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
