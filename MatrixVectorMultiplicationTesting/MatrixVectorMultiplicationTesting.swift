//
//  MatrixVectorMultiplicationTesting.swift
//  MatrixVectorMultiplicationTesting
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import XCTest

final class MatrixVectorMultiplicationTesting: XCTestCase {
    private var matricesAndVectors : [(Matrix, Vector)] = []
    private var resullts : [Vector] = [
        Vector(data: [16, -3, 14]),
        Vector(data: [-5, 12]),
        Vector(data: [11, -8, 16]),
        Vector(data: [-15, 17, 39])
    ]
    
    private var tranposedResult : [Matrix] = [
        Matrix(data: [
            [3, -2, 2],
            [5, -4, 5],
            [1, 1, 1]
        ]),
        Matrix(data: [
            [3, -2],
            [5, -4],
            [1, 1]
        ])
    ]
        
    override func setUpWithError() throws {
        matricesAndVectors.append(
            (Matrix(data: [
                [3, 5, 1],
                [-2, -4, 1],
                [2, 5, 1]
            ]), Vector(data: [2, 1, 5]))
        )
        
        matricesAndVectors.append(
            (Matrix(data: [
                [3, 5, 1],
                [-2, -4, 1],
            ]), Vector(data: [2, -3, 4]))
        )
        
        matricesAndVectors.append(
            (Matrix(data: [
                [3, 5],
                [-2, -4],
                [5, 6]
            ]), Vector(data: [2, 1]))
        )
        
        matricesAndVectors.append(
            (Matrix(data: [
                [3, 5, -4, -6],
                [-2, -4, 5, 0],
                [5, 1, 5, 3]
            ]), Vector(data: [2, 1, 5, 1]))
        )
        
        
    }

    func testMultiplication() throws {
        do {
            for i in (0..<(matricesAndVectors.count)) {
                let (mat, vec) = matricesAndVectors[i]
                let result = try (mat * vec)
                XCTAssertTrue(result.identical(resullts[i]))
            }
        }
        catch _ {
            
        }
    }
    
    func testTranspose() {
        let result1 = matricesAndVectors[0].0.transpose()
        let result2 = matricesAndVectors[1].0.transpose()
        
        XCTAssertTrue(result1.identical(mat: tranposedResult[0]))
        XCTAssertTrue(result2.identical(mat: tranposedResult[1]))
    }
    
    func testVectorization() {
        let data = Matrix(data: [
            [3, 5, 6, 1]
        ])
        
        let data2 = Matrix(data: [
            [2], [7], [6]
        ])
        
        XCTAssertTrue(try data.vectorized().identical(Vector(data: [3, 5, 6, 1])))
        XCTAssertTrue(try data2.vectorized().identical(Vector(data: [2, 7, 6])))
    }
    
    func testMatrixAddition() {
        let mat1 = Matrix(data: [
            [3, 5, 9],
            [6, 2, 1]
        ])
        
        let mat2 = Matrix(data: [
            [-3, 5, 8],
            [4, 2, 0]
        ])
        
        XCTAssertTrue((mat1 + mat2).identical(mat: Matrix(data: [
            [0, 10, 17],
            [10, 4, 1]
        ])))
    }
    
    func testScaledMatrix() {
        let mat = Matrix(data: [
            [3, 5, 2],
            [4, 2, 1]
        ])
        
        XCTAssertTrue(mat.scaled(factor: 2.0)
            .identical(mat: Matrix(data: [
                [6, 10, 4],
                [8, 4, 2]
            ])))
    }
}
