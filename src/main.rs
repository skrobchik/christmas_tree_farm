use std::str::FromStr;

use itertools::Itertools;
use rustsat::solvers::Solve;
use rustsat::types::Assignment;
use rustsat::{
    instances::SatInstance,
    solvers::SolverResult,
    types::{Lit, constraints::CardConstraint},
};

const PRESENT_SIZE: usize = 3;

#[derive(Debug, Clone, Default)]
struct PresentShape([[bool; PRESENT_SIZE]; PRESENT_SIZE]);

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
            let mut shape = [[false; PRESENT_SIZE]; PRESENT_SIZE];
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

struct SolutionFormatter<'a> {
    shapes: &'a [PresentShape],
    query: &'a Query,
    shape_placed: &'a ndarray::Array3<Lit>,
    solution: &'a Assignment,
}

impl<'a> std::fmt::Display for SolutionFormatter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows = PRESENT_SIZE + self.query.rows; // "fake" rows and columns
        let cols = PRESENT_SIZE + self.query.cols;
        let mut m: ndarray::Array2<Option<usize>> = ndarray::Array::default((rows, cols));
        for (i_shape, shape) in self.shapes.iter().enumerate() {
            for (i, j) in (0..rows).cartesian_product(0..cols) {
                match self.solution.lit_value(self.shape_placed[(i_shape, i, j)]) {
                    rustsat::types::TernaryVal::False | rustsat::types::TernaryVal::DontCare => {
                        continue;
                    }
                    rustsat::types::TernaryVal::True => (),
                }
                write!(f, "shape {} place at ({}, {})\n", i_shape, i, j)?;
                for (di, dj) in (0..PRESENT_SIZE).cartesian_product(0..PRESENT_SIZE) {
                    if !shape.0[di][dj] {
                        continue;
                    }
                    m[(i + di, j + dj)] = Some(i_shape);
                }
            }
        }
        for (i, j) in (0..rows).cartesian_product(0..cols) {
            match m[(i, j)] {
                Some(i_shape) => {
                    assert!(i_shape < 10);
                    write!(f, "{}", i_shape)?;
                }
                None => {
                    write!(f, ".")?;
                }
            }
            if j + 1 == cols && i + 1 != rows {
                write!(f, "\n")?;
            }
        }
        Ok(())
    }
}

#[must_use]
fn rotate_shape_clockwise(shape: &PresentShape) -> PresentShape {
    let mut shape1 = [[false; PRESENT_SIZE]; PRESENT_SIZE];
    for ring in 0..=PRESENT_SIZE - 2 {
        for i in 0..PRESENT_SIZE - 2 * ring {
            let start = ring;
            let end = PRESENT_SIZE - ring - 1;
            shape1[start + i][end] = shape.0[start][start + i];
            shape1[end][end - i] = shape.0[start + i][end];
            shape1[end - i][start] = shape.0[end][end - i];
            shape1[start][start + i] = shape.0[end - i][start];
        }
    }
    PresentShape(shape1)
}

fn solve(query: &Query, shapes: &[PresentShape], test_case: usize) -> bool {
    let rows = PRESENT_SIZE + query.rows;
    let cols = PRESENT_SIZE + query.cols;
    let mut instance: SatInstance = SatInstance::new();

    let mut shapes: Vec<PresentShape> = shapes.into();
    let num_shapes = shapes.len();
    shapes.resize(num_shapes * 4, PresentShape::default());
    for i in num_shapes..shapes.len() {
        shapes[i] = rotate_shape_clockwise(&shapes[i - num_shapes]);
    }
    let shapes = shapes;

    let shape_placed: ndarray::Array3<Lit> =
        ndarray::Array::from_shape_simple_fn((shapes.len(), rows, cols), || instance.new_lit()); // [shape, row, col]

    // Present Quantity Requirements
    for shape_index in 0..num_shapes {
        let mut literals: Vec<Lit> = Vec::with_capacity(4 * rows * cols);
        for (i, j) in (0..rows).cartesian_product(0..cols) {
            literals.push(shape_placed[(shape_index + 0 * num_shapes, i, j)]);
            literals.push(shape_placed[(shape_index + 1 * num_shapes, i, j)]);
            literals.push(shape_placed[(shape_index + 2 * num_shapes, i, j)]);
            literals.push(shape_placed[(shape_index + 3 * num_shapes, i, j)]);
        }
        instance.add_card_constr(CardConstraint::new_eq(
            literals,
            query.present_requirements[shape_index],
        ));
    }

    // Present Shape Geometry Constraints
    for (i, j) in (0..rows).cartesian_product(0..cols) {
        let mut literals: Vec<Lit> = Vec::new();
        for (shape_index, (i_shape, j_shape)) in (0..shapes.len()).cartesian_product(
            (i.saturating_sub(PRESENT_SIZE - 1)..=i)
                .cartesian_product(j.saturating_sub(PRESENT_SIZE - 1)..=j),
        ) {
            if shapes[shape_index].0[i - i_shape][j - j_shape] {
                literals.push(shape_placed[(shape_index, i_shape, j_shape)]);
            }
        }
        instance.add_card_constr(CardConstraint::new_eq(
            literals,
            if i >= PRESENT_SIZE && j >= PRESENT_SIZE {
                1
            } else {
                0
            },
        ));
    }
    let mut instance = instance.sanitize();
    instance.convert_to_cnf();
    let mut file =
        std::fs::File::create(format!("christmas_tree_farm_{}.dimacs", test_case)).unwrap();
    instance.write_dimacs(&mut file).unwrap();
    let mut solver = rustsat_batsat::BasicSolver::default();
    solver.add_cnf(instance.into_cnf().0).unwrap();
    if matches!(solver.solve(), Ok(SolverResult::Sat)) {
        let solution = solver.full_solution().unwrap();
        println!(
            "{}",
            SolutionFormatter {
                shapes: &shapes,
                query,
                shape_placed: &shape_placed,
                solution: &solution
            }
        );
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
