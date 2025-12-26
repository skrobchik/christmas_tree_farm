use std::str::FromStr;

use itertools::Itertools;
use rustsat::solvers::Solve;
use rustsat::{
    encodings::CollectClauses,
    instances::SatInstance,
    solvers::SolverResult,
    types::{Lit, constraints::CardConstraint},
};

#[derive(Debug)]
struct PresentShape([[bool; 3]; 3]);

#[derive(Debug)]
struct Query {
    rows: usize,
    cols: usize,
    present_requirements: Vec<usize>,
}

impl FromStr for Query {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (dimension, requirements) = s.split_once(':').ok_or("expected ':''")?;
        let (cols, rows) = dimension.split_once('x').ok_or("expected 'x'")?;
        let rows = rows.parse()?;
        let cols = cols.parse()?;
        let present_requirements = requirements
            .split_whitespace()
            .map(|s| s.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Query {
            rows,
            cols,
            present_requirements,
        })
    }
}

fn parse_input(input: &str) -> Result<(Vec<PresentShape>, Vec<Query>), Box<dyn std::error::Error>> {
    let lines = input.lines().map(str::trim);
    let line_groups = lines.chunk_by(|s| str::is_empty(s));
    let mut line_groups = line_groups.into_iter();
    let mut queries = Vec::new();
    let mut shapes = Vec::new();
    while let Some((key, mut group)) = line_groups.next() {
        if key {
            continue;
        }
        let first_line = match group.next() {
            Some(x) => x,
            None => continue,
        };
        let is_shape = first_line.ends_with(":");
        if is_shape {
            let mut shape = [[false; 3]; 3];
            let mut i = 0;
            while let Some(line) = group.next() {
                for (j, c) in line.chars().enumerate() {
                    shape[i][j] = c == '#';
                }
                i += 1;
            }
            shapes.push(PresentShape(shape));
        } else {
            queries.push(Query::from_str(first_line)?);
            while let Some(line) = group.next() {
                queries.push(Query::from_str(line)?);
            }
        }
    }
    Ok((shapes, queries))
}

fn solve(query: &Query, shapes: &[PresentShape], test_case: usize) -> bool {
    let rows = query.rows;
    let cols = query.cols;
    let mut instance: SatInstance = SatInstance::new();
    let shape_placed: ndarray::Array3<Lit> =
        ndarray::Array::from_shape_simple_fn((shapes.len(), rows, cols), || instance.new_lit());
    let occupied_by_shape: ndarray::Array4<Lit> =
        ndarray::Array::from_shape_simple_fn((rows, cols, rows, cols), || instance.new_lit());
    // No place can be occupied by more than one shape
    for (i, j) in (0..rows).cartesian_product(0..cols) {
        let mut constraint = CardConstraint::new_eq([], 1);
        for (i_shape, j_shape) in (0..rows).cartesian_product(0..cols) {
            constraint.add([occupied_by_shape[(i_shape, j_shape, i, j)]]);
        }
        instance.add_card_constr(constraint);
    }
    // Shape geometries
    for (i_shape, shape) in shapes.iter().enumerate() {
        for (i, j) in (0..rows).cartesian_product(0..cols) {
            for (di, dj) in (0..3).cartesian_product(0..3) {
                if !shape.0[di][dj] {
                    continue;
                }
                if i + di < rows && j + dj < cols {
                    instance.add_lit_impl_lit(
                        shape_placed[(i_shape, i, j)],
                        occupied_by_shape[(i, j, i + di, j + dj)],
                    );
                } else {
                    // Shape would be out of bounds
                    instance.add_card_constr(CardConstraint::new_eq(
                        [shape_placed[(i_shape, i, j)]],
                        0,
                    ));
                }
            }
        }
    }
    // Required placed shapes
    for i_shape in 0..shapes.len() {
        let mut constraint = CardConstraint::new_lb([], query.present_requirements[i_shape]);
        for (i, j) in (0..rows).cartesian_product(0..cols) {
            constraint.add([shape_placed[(i_shape, i, j)]]);
        }
        instance.add_card_constr(constraint);
    }
    instance.convert_to_cnf();
    let mut file =
        std::fs::File::create(format!("christmas_tree_farm_{}.dimacs", test_case)).unwrap();
    instance.write_dimacs(&mut file).unwrap();
    let mut solver = rustsat_batsat::BasicSolver::default();
    solver.add_cnf(instance.into_cnf().0).unwrap();
    if matches!(solver.solve(), Ok(SolverResult::Sat)) {
        println!("{:?}", solver.full_solution());
        true
    } else {
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = std::fs::read_to_string("input.txt")?;
    let (shapes, queries) = parse_input(&input)?;
    for (test_case, query) in queries.iter().enumerate() {
        if solve(&query, &shapes, test_case) {
            println!("Case #{}: YES", test_case)
        } else {
            println!("Case #{}: NO", test_case)
        }
    }
    Ok(())
}
