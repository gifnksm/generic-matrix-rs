// Copyright 2016 generic-matrix-rs Developers
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Manipulations and data types that represent 2d matrix.

#![warn(bad_style, missing_docs, unused, unused_extern_crates, unused_import_braces,
       unused_qualifications, unused_results)]

extern crate num_traits;

use std::mem::swap;
use std::ops::{Add, Index, IndexMut, Mul, Sub};
use num_traits::{One, Zero};

/// 2D matrix.
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct Matrix<T> {
    row: usize,
    column: usize,
    data: Vec<T>,
}

impl<T> Matrix<T> {
    /// Creates a new `Matrix`.
    #[inline]
    pub fn from_fn<F>(row: usize, column: usize, f: F) -> Matrix<T>
    where
        F: Fn(usize, usize) -> T,
    {
        Matrix {
            row: row,
            column: column,
            data: (0..row * column)
                .map(|i| f(i / column, i % column))
                .collect(),
        }
    }

    /// Creates a new `Matrix` from vector.
    #[inline]
    pub fn from_vec(row: usize, column: usize, data: Vec<T>) -> Matrix<T> {
        assert_eq!(row * column, data.len());
        Matrix {
            row: row,
            column: column,
            data: data,
        }
    }

    /// Returns the matrix's row and column.
    #[inline]
    pub fn size(&self) -> (usize, usize) {
        (self.row(), self.column())
    }
    /// Returns the matrix's row.
    #[inline]
    pub fn row(&self) -> usize {
        self.row
    }
    /// Returns the matrix's column.
    #[inline]
    pub fn column(&self) -> usize {
        self.column
    }
    
    /// Transposes the matrix in-place.
    pub fn trans_in_place(&mut self) {
        if self.row == self.column {
            // easy case of square matrix
            for i in 0..self.row {
                for j in 0..i {
                    self.data.swap(i * self.column + j, j * self.column + i);
                }
            }
        } else {
            // easy case of either dimension being zero or one
            swap(&mut self.row, &mut self.column);
            if self.row > 1 && self.column > 1 {
                // hard case of non-square matrix with both dimensions at least two
                let mut skip_bitmap = vec![0u32; (self.row * self.column + 31) / 32];
                
                for i in 0..self.row {
                    for j in 0..self.column {
                        // within this block is where bugs are most likely to be
                        let original_this = i * self.column + j;
                        let mut this = original_this;
                        let mut other = j * self.row + i;
                        // make sure each rotation is performed exactly once
                        while original_this < other && skip_bitmap[this / 32] & (1u32 << (this % 32)) == 0 {
                            self.data.swap(this, other);
                            skip_bitmap[this / 32] |= 1u32 << (this % 32);
                            this = other;
                            other = (this % self.column) * self.row + (this / self.column);
                        }
                    }
                }
            }
        }
    }
}

impl<T: Zero> Matrix<T> {
    /// Creates a matrix whose elements are all zero.
    #[inline]
    pub fn zero(row: usize, column: usize) -> Matrix<T> {
        Matrix::from_fn(row, column, |_, _| Zero::zero())
    }
}

impl<T: One + Zero> Matrix<T> {
    /// Creates a identity matrix.
    #[inline]
    pub fn one(row: usize, column: usize) -> Matrix<T> {
        Matrix::from_fn(row, column, |i, j| if i == j {
            One::one()
        } else {
            Zero::zero()
        })
    }
}

impl<T: Clone> Matrix<T> {
    #[inline]
    /// Returns transpose of the matrix.
    pub fn trans(&self) -> Matrix<T> {
        Matrix::from_fn(self.column(), self.row(), |i, j| self[(j, i)].clone())
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        assert!(i < self.row() && j < self.column());
        &self.data[i * self.column() + j]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        assert!(i < self.row && j < self.column);
        &mut self.data[i * self.column + j]
    }
}

macro_rules! forward_val_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<Lhs, Rhs> $imp<Matrix<Rhs>> for Matrix<Lhs>
            where Lhs: $imp<Rhs> + Clone, Rhs: Clone
        {
            type Output = Matrix<<Lhs as $imp<Rhs>>::Output>;

            #[inline]
            fn $method(self, other: Matrix<Rhs>) -> Matrix<<Lhs as $imp<Rhs>>::Output> {
                $imp::$method(&self, &other)
            }
        }
    }
}

macro_rules! forward_ref_val_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, Lhs, Rhs> $imp<Matrix<Rhs>> for &'a Matrix<Lhs>
            where Lhs: $imp<Rhs> + Clone, Rhs: Clone
        {
            type Output = Matrix<<Lhs as $imp<Rhs>>::Output>;

            #[inline]
            fn $method(self, other: Matrix<Rhs>) -> Matrix<<Lhs as $imp<Rhs>>::Output> {
                $imp::$method(self, &other)
            }
        }
    }
}

macro_rules! forward_val_ref_binop {
    (impl $imp:ident, $method:ident) => {
        impl<'a, Lhs, Rhs> $imp<&'a Matrix<Rhs>> for Matrix<Lhs>
            where Lhs: $imp<Rhs> + Clone, Rhs: Clone
        {
            type Output = Matrix<<Lhs as $imp<Rhs>>::Output>;

            #[inline]
            fn $method(self, other: &Matrix<Rhs>) -> Matrix<<Lhs as $imp<Rhs>>::Output> {
                $imp::$method(&self, other)
            }
        }
    }
}

macro_rules! forward_all_binop {
    (impl $imp:ident, $method:ident) => {
        forward_val_val_binop!(impl $imp, $method);
        forward_ref_val_binop!(impl $imp, $method);
        forward_val_ref_binop!(impl $imp, $method);
    };
}

forward_all_binop!(impl Add, add);

impl<'a, 'b, Lhs, Rhs> Add<&'b Matrix<Rhs>> for &'a Matrix<Lhs>
where
    Lhs: Add<Rhs> + Clone,
    Rhs: Clone,
{
    type Output = Matrix<<Lhs as Add<Rhs>>::Output>;

    #[inline]
    fn add(self, other: &Matrix<Rhs>) -> Matrix<<Lhs as Add<Rhs>>::Output> {
        assert_eq!(self.size(), other.size());
        Matrix::from_fn(self.row(), self.column(), |i, j| {
            self[(i, j)].clone() + other[(i, j)].clone()
        })
    }
}

forward_all_binop!(impl Sub, sub);

impl<'a, 'b, Lhs, Rhs> Sub<&'b Matrix<Rhs>> for &'a Matrix<Lhs>
where
    Lhs: Sub<Rhs> + Clone,
    Rhs: Clone,
{
    type Output = Matrix<<Lhs as Sub<Rhs>>::Output>;

    #[inline]
    fn sub(self, other: &Matrix<Rhs>) -> Matrix<<Lhs as Sub<Rhs>>::Output> {
        assert_eq!(self.size(), other.size());
        Matrix::from_fn(self.row(), self.column(), |i, j| {
            self[(i, j)].clone() - other[(i, j)].clone()
        })
    }
}

impl<Lhs, Rhs> Mul<Matrix<Rhs>> for Matrix<Lhs>
where
    Lhs: Mul<Rhs> + Clone,
    Rhs: Clone,
    <Lhs as Mul<Rhs>>::Output: Add<Output = <Lhs as Mul<Rhs>>::Output>,
{
    type Output = Matrix<<Lhs as Mul<Rhs>>::Output>;

    #[inline]
    fn mul(self, other: Matrix<Rhs>) -> Matrix<<Lhs as Mul<Rhs>>::Output> {
        Mul::mul(&self, &other)
    }
}

impl<'a, Lhs, Rhs> Mul<Matrix<Rhs>> for &'a Matrix<Lhs>
where
    Lhs: Mul<Rhs> + Clone,
    Rhs: Clone,
    <Lhs as Mul<Rhs>>::Output: Add<Output = <Lhs as Mul<Rhs>>::Output>,
{
    type Output = Matrix<<Lhs as Mul<Rhs>>::Output>;

    #[inline]
    fn mul(self, other: Matrix<Rhs>) -> Matrix<<Lhs as Mul<Rhs>>::Output> {
        Mul::mul(self, &other)
    }
}

impl<'a, Lhs, Rhs> Mul<&'a Matrix<Rhs>> for Matrix<Lhs>
where
    Lhs: Mul<Rhs> + Clone,
    Rhs: Clone,
    <Lhs as Mul<Rhs>>::Output: Add<Output = <Lhs as Mul<Rhs>>::Output>,
{
    type Output = Matrix<<Lhs as Mul<Rhs>>::Output>;

    #[inline]
    fn mul(self, other: &Matrix<Rhs>) -> Matrix<<Lhs as Mul<Rhs>>::Output> {
        Mul::mul(&self, other)
    }
}

impl<'a, 'b, Lhs, Rhs> Mul<&'b Matrix<Rhs>> for &'a Matrix<Lhs>
where
    Lhs: Mul<Rhs> + Clone,
    Rhs: Clone,
    <Lhs as Mul<Rhs>>::Output: Add<Output = <Lhs as Mul<Rhs>>::Output>,
{
    type Output = Matrix<<Lhs as Mul<Rhs>>::Output>;

    #[inline]
    fn mul(self, other: &Matrix<Rhs>) -> Matrix<<Lhs as Mul<Rhs>>::Output> {
        assert_eq!(self.column(), other.row());
        Matrix::from_fn(self.row(), other.column(), |i, j| {
            let mut sum = self[(i, 0)].clone() * other[(0, j)].clone();
            for k in 1..self.column() {
                sum = sum + self[(i, k)].clone() * other[(k, j)].clone();
            }
            sum
        })
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn from_vec() {
        let mat = Matrix::from_vec(2, 3, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]);
        for i in 0..mat.row() {
            for j in 0..mat.column() {
                assert_eq!((i, j), mat[(i, j)]);
            }
        }
    }

    #[test]
    fn index() {
        let mat = Matrix::from_fn(3, 5, |i, j| (i, j));
        for i in 0..mat.row() {
            for j in 0..mat.column() {
                assert_eq!((i, j), mat[(i, j)]);
            }
        }
    }

    #[test]
    fn index_mut() {
        let mut m = Matrix::one(2, 2);
        m[(1, 1)] = 0;
        assert_eq!(Matrix::from_vec(2, 2, vec![1, 0, 0, 0]), m);
    }

    #[test]
    fn mul() {
        let m1 = Matrix::from_vec(1, 3, vec![1.0f64, 2.0, 3.0]);
        let m2 = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
        assert_eq!(Matrix::from_vec(1, 1, vec![14.0]), m1 * m2);
        assert_eq!(
            Matrix::from_vec(3, 1, vec![1.0f64, 4.0, 7.0]),
            Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                * Matrix::from_vec(3, 1, vec![1.0, 0.0, 0.0])
        );
        assert_eq!(
            Matrix::from_vec(3, 2, vec![1.0f64, 3.0, 4.0, 6.0, 7.0, 9.0]),
            Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                * Matrix::from_vec(3, 2, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        );
    }
    
    #[test]
    fn trans() {
        let mut square = Matrix::from_vec(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(square.trans(), Matrix::from_vec(3, 3, vec![1, 4, 7, 2, 5, 8, 3, 6, 9]));
        square.trans_in_place();
        assert_eq!(square, Matrix::from_vec(3, 3, vec![1, 4, 7, 2, 5, 8, 3, 6, 9]));
        
        let mut vector = Matrix::from_vec(3, 1, vec![1, 2, 3]);
        assert_eq!(vector.trans(), Matrix::from_vec(1, 3, vec![1, 2, 3]));
        vector.trans_in_place();
        assert_eq!(vector, Matrix::from_vec(1, 3, vec![1, 2, 3]));
        
        let mut rect_2_3 = Matrix::from_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(rect_2_3.trans(), Matrix::from_vec(2, 3, vec![1, 3, 5, 2, 4, 6]));
        rect_2_3.trans_in_place();
        assert_eq!(rect_2_3, Matrix::from_vec(2, 3, vec![1, 3, 5, 2, 4, 6]));
        
        let mut rect_5_2 = Matrix::from_vec(2, 5, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(rect_5_2.trans(), Matrix::from_vec(5, 2, vec![1, 6, 2, 7, 3, 8, 4, 9, 5, 10]));
        rect_5_2.trans_in_place();
        assert_eq!(rect_5_2, Matrix::from_vec(5, 2, vec![1, 6, 2, 7, 3, 8, 4, 9, 5, 10]));
        
        let mut rect_5_3 = Matrix::from_vec(3, 5, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(rect_5_3.trans(), Matrix::from_vec(5, 3, vec![1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]));
        rect_5_3.trans_in_place();
        assert_eq!(rect_5_3, Matrix::from_vec(5, 3, vec![1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]));
    }
}
