//
//  CostFunction.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import Foundation

enum CrossEntropyError: Error {
    case CORRUPTED_LABEL
}

protocol CostFunction {
    func errorOnDataset(a: Vector, label:Vector) throws -> Double
    func derivativeAt(a: Vector, label: Vector) throws -> Vector
}

class CrossEntropy : CostFunction {
    func errorOnDataset(a: Vector, label: Vector) throws -> Double {
        for vecIndex in (0..<label.size) {
            if label[vecIndex] == 1 {
                return -log(a[vecIndex] + 1e-8)
            }
        }
        
        throw CrossEntropyError.CORRUPTED_LABEL
    }
    
    
    func derivativeAt(a: Vector, label: Vector) throws -> Vector {
        var result = Vector(label.size)
        
        for vecIndex in (0..<label.size) {
            if label[vecIndex] == 1 {
                result[vecIndex] = -1 / (a[vecIndex] + 1e-8)
                return result
            }
        }
        
        throw CrossEntropyError.CORRUPTED_LABEL
    }
}

class MSE : CostFunction {
    func errorOnDataset(a: Vector, label: Vector) throws -> Double {
        var error = 0.0
        
        for i in (0..<label.size) {
            let term = a[i] - label[i]
            error += term * term
        }
        
        return error / 2
    }
    
    func derivativeAt(a: Vector, label: Vector) throws -> Vector {
        var result = Vector(label.size)
        
        for i in (0..<result.size) {
            result[i] = a[i] - label[i]
        }
        
        return result
    }
}
