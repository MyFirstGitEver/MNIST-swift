//
//  Matrix.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import Foundation

enum MatrixError : Error {
    case WRONG_SIZE
    case CANT_VECTORIZE
}

struct Matrix {
    private var entries : [[Double]] = []
    
    init(firstDimension: Int, secondDimension: Int) {
        entries.reserveCapacity(firstDimension)
        
        for rowIndex in (0..<firstDimension) {
            entries.append([])
            entries[rowIndex].reserveCapacity(secondDimension)
            
            for _ in (0..<secondDimension) {
                entries[rowIndex].append(Double.random(in: 0...1))
            }
        }
    }
    
    init(data: [[Double]]) {
        entries = data
    }
    
    func getShape() -> (Int, Int) {
        return (entries.count, entries[0].count)
    }
    
    func transpose() -> Matrix {
        let m = entries.count
        let n = entries[0].count
        
        var newMatrix = Matrix(firstDimension: n, secondDimension: m)
        
        for i in (0..<n) {
            for j in (0..<m) {
                newMatrix.entries[i][j] = entries[j][i]
            }
        }
        
        return newMatrix
    }
    
    mutating func setEntryAt(i : Int, j: Int, value: Double) {
        entries[i][j] = value
    }
    
    func vectorized() throws -> Vector {
        let m = entries.count
        let n = entries[0].count
        
        if (n != 1 && m != 1) {
            throw MatrixError.CANT_VECTORIZE
        }
        
        if m == 1 {
            var result = Vector(n)
            
            for i in (0..<n) {
                result[i] = entries[0][i]
            }
            
            return result
        }
        
        var result = Vector(m)
        
        for i in (0..<m) {
            result[i] = entries[i][0]
        }
        
        return result
    }
    
    func identical(mat: Matrix) -> Bool {
        let shape = mat.getShape()
        let m = shape.0
        let n = shape.1
        
        if m != entries.count || n != entries[0].count {
            return false
        }
        
        for i in (0..<m) {
            for j in (0..<n) {
                if entries[i][j] != mat.entries[i][j] {
                    return false
                }
            }
        }
        
        return true
    }
    
    func scaled(factor: Double) -> Matrix {
        let (m, n) = getShape()
        
        var scaledMatrix = Matrix(firstDimension: m, secondDimension: n)
        
        for i in (0..<m) {
            for j in (0..<n) {
                scaledMatrix.entries[i][j] = factor * entries[i][j]
            }
        }
        
        return scaledMatrix
    }
    
    static func +(mat1: Matrix, mat2: Matrix) -> Matrix {
        let shape = mat1.getShape()
        let m = shape.0
        let n = shape.1
        
        var result = Matrix(firstDimension: m, secondDimension: n)
        for i in (0..<m) {
            for j in (0..<n) {
                result.entries[i][j] =
                    mat1.entries[i][j] + mat2.entries[i][j]
            }
        }
        
        return result
    }
    
    static func *(mat: Matrix, vec: Vector) throws -> Vector {
        let shape = mat.getShape()
        let first = shape.0
        let second = shape.1
        
        if second != vec.size {
            throw MatrixError.WRONG_SIZE
        }
        
        var resultVector = Vector(first)
        for rowIndex in (0..<first) {
            var entryOfResultVector = 0.0
            for columnIndex in (0..<second) {
                entryOfResultVector +=
                    (mat.entries[rowIndex][columnIndex] * vec[columnIndex])
            }
            
            resultVector[rowIndex] = entryOfResultVector
        }
        
        return resultVector
    }
}
