//
//  Matrix.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import Foundation

enum MatrixError : Error {
    case WRONG_SIZE
}

struct Matrix : Codable {
    private var entries : [[Double]] = []
    
    init(firstDimension: Int, secondDimension: Int) {
        entries.reserveCapacity(firstDimension)
        
        for rowIndex in (0..<firstDimension) {
            entries.append([])
            entries[rowIndex].reserveCapacity(secondDimension)
            
            for _ in (0..<secondDimension) {
                entries[rowIndex].append(0)
            }
        }
    }
    
    init(data: [[Double]]) {
        entries = data
    }
    
    mutating func randomise() {
        let (m, n) = getShape()
        
        for i in (0..<m) {
            for j in (0..<n) {
                entries[i][j] = Double.random(in: 0...1) * 0.01
            }
        }
    }
    
    func getShape() -> (Int, Int) {
        return (entries.count, entries[0].count)
    }
    
    mutating func setEntryAt(i : Int, j: Int, value: Double) {
        entries[i][j] = value
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
    
    func squared() -> Matrix {
        let (m, n) = getShape()
        
        var squaredMatrix = Matrix(firstDimension: m, secondDimension: n)
        
        for i in (0..<m) {
            for j in (0..<n) {
                squaredMatrix.entries[i][j] = entries[i][j] * entries[i][j]
            }
        }
        
        return squaredMatrix
    }
    
    func sqrt() -> Matrix {
        let (m, n) = getShape()
        
        var rootedMatrix = Matrix(firstDimension: m, secondDimension: n)
        
        for i in (0..<m) {
            for j in (0..<n) {
                rootedMatrix.entries[i][j] = entries[i][j].squareRoot()
            }
        }
        
        return rootedMatrix
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
    
    static func -(mat1: Matrix, mat2: Matrix) -> Matrix{
        let shape = mat1.getShape()
        let m = shape.0
        let n = shape.1
        
        var result = Matrix(firstDimension: m, secondDimension: n)
        for i in (0..<m) {
            for j in (0..<n) {
                result.entries[i][j] =
                    mat1.entries[i][j] - mat2.entries[i][j]
            }
        }
        
        return result
    }
    
    static func /(mat1: Matrix, mat2: Matrix) -> Matrix {
        let shape = mat1.getShape()
        let m = shape.0
        let n = shape.1
        
        var result = Matrix(firstDimension: m, secondDimension: n)
        for i in (0..<m) {
            for j in (0..<n) {
                result.entries[i][j] =
                    mat1.entries[i][j] / (mat2.entries[i][j] + 1e-8)
            }
        }
        
        return result
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
}
