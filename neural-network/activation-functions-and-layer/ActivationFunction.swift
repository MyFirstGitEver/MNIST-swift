//
//  ActivationFunction.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import Foundation

protocol ActivationFunction {
    func f(z: Vector) -> Vector
    func derivative(x: Vector, index: Int) -> Vector
}

class ReluActivation : ActivationFunction {
    func derivative(x: Vector, index: Int) -> Vector {
        var result = Vector(x.size)
        let a = f(z: x)
        
        for i in (0..<(x.size)) {
            if i == index {
                result[i] = (a[i] <= 0 ? 0 : 1)
                break
            }
        }
        
        return result
    }
    
    func f(z: Vector) -> Vector {
        var result = Vector(z.size)
        
        for i in (0..<(z.size)) {
            result[i] = max(z[i], 0)
        }
        
        return result
    }
}

class SoftmaxActivation : ActivationFunction {
    func derivative(x: Vector, index: Int) -> Vector {
        var result = Vector(x.size)
        let a = f(z: x)
        
        for i in (0..<result.size) {
            if i == index {
                result[i] = a[index] * (1 - a[index])
            }
            else {
                result[i] = (-a[index] * a[i])
            }
        }
        
        return result
    }
    
    func f(z: Vector) -> Vector {
        var maxEntryInVec = -Double.greatestFiniteMagnitude
        
        for i in (0..<z.size) {
            maxEntryInVec = max(maxEntryInVec, z[i])
        }
        
        var below = 0.0
        
        for i in (0..<z.size) {
            below += exp(z[i] - maxEntryInVec)
        }
        
        var result = Vector(z.size)
        
        for i in (0..<z.size) {
            result[i] = exp(z[i] - maxEntryInVec) / below
        }
        
        return result
    }
}
