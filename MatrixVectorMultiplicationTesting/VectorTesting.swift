//
//  RowMulVector.swift
//  MatrixVectorMultiplicationTesting
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import XCTest

final class RowMulVectorTesting: XCTestCase {
    private var pairs : [(Vector, Vector)] = []
    
    private var results = [
        Matrix(data: [
            [9, 18, 0],
            [15, 30, 0],
            [12, 24, 0],
            [6, 12, 0],
            [-3, -6, 0],
        ]),
        Matrix(data: [
            [12, 30],
            [-6, -15],
            [8, 20],
            [-2, -5]
        ]),
        Matrix(data: [
            [9, -15, 0],
            [-12, 20, 0],
            [24, -40, 0],
            [36, -60, 0],
            [0, 0, 0]
        ]),
        Matrix(data: [
            [18],
            [30],
            [24]
        ])
    ]
    
    override func setUpWithError() throws {
        pairs.append((
            Vector(data: [3, 5, 4, 2, -1]),
            Vector(data: [3, 6, 0])
        ))
        
        pairs.append((
            Vector(data: [6, -3, 4, -1]),
            Vector(data: [2, 5])
        ))
        
        pairs.append((
            Vector(data: [3, -4, 8, 12, 0]),
            Vector(data: [3, -5, 0])
        ))
        
        pairs.append((
            Vector(data: [3, 5, 4]),
            Vector(data: [6])
        ))
    }

    func testRowMul() throws {
        for (index, pair) in pairs.enumerated() {
            let matrix = pair.0.mulWithRowVector(v: pair.1)
            XCTAssertTrue(matrix.identical(mat: results[index]))
        }
    }
    
    func testConcat() {
        var v = Vector(data: [3, 5, 1, 6])
        v.concat(v: Vector(data: [5, 3, 0]))
        
        XCTAssertTrue(v.identical(Vector(data: [3, 5, 1, 6, 5, 3, 0])))
    }
    
    func testNormalise() {
        var v1 = Vector(data: [2, 0])
        var v2 = Vector(data: [2, 1, 2])
        
        v1.normalize()
        v2.normalize()
        
        XCTAssertTrue(v1.identical(Vector(data: [1, 0])))
        XCTAssertTrue(v2.identical(Vector(data: [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0])))
    }
}
