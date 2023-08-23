//
//  Vector.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import Foundation

enum VectorError : Error {
    case VEC_WRONG_SIZE
}

struct Vector : Codable {
    private var points : [Double]

    var size: Int {
        points.count
    }

    init(_ size : Int) {
        points = Array(
            repeating: 0, count: size)
    }

    init(data: [Double]) {
        points = data
    }
    
    mutating func randomise() {
        for i in 0..<size {
            points[i] = Double.random(in: 0...1)
        }
    }
    
    mutating func scaleBy(_ factor : Double) {
        for i in 0..<size {
            points[i] *= factor
        }
    }
    
    mutating func subtract(_ v : Vector) {
        for i in 0..<size {
            points[i] -= v[i]
        }
    }
    
    func distTo(to: Vector) -> Double {
        var dist = 0.0
        let vecDimension = to.points.count
        
        for i in (0..<vecDimension) {
            let term = (points[i] - to.points[i])
            dist += (term * term)
        }
        
        return dist.squareRoot()
    }
    
    func mulWithRowVector(v : Vector) -> Matrix {
        var result = Matrix(firstDimension: size, secondDimension: v.size)
        let m = size
        let n = v.size
        
        for j in (0..<n) {
            for i in (0..<m) {
                result.setEntryAt(i: i, j: j, value: points[i] * v[j])
            }
        }
        
        return result
    }
    
    func sum() -> Double {
        var total = 0.0
        
        for entry in points {
            total += entry
        }
        
        return total
    }
    
    func hadamard(v: Vector) throws -> Vector {
        if v.size != points.count {
            throw VectorError.VEC_WRONG_SIZE
        }
        
        var result = Vector(v.size)
        
        for i in (0..<v.size) {
            result[i] = v[i] * points[i]
        }
        
        return result
    }
    
    func sqrt() -> Vector {
        var v = Vector(size)
        
        for i in (0..<size) {
            v[i] = points[i].squareRoot()
        }
        
        return v
    }
    
    func scaled(factor: Double) -> Vector {
        var v = Vector(size)
        
        for i in (0..<size) {
            v[i] = factor * points[i]
        }
        
        return v
    }
    
    func squared() -> Vector {
        var v = Vector(size)
        
        for i in (0..<size) {
            v[i] = points[i] * points[i]
        }
        
        return v
    }
 
    static func *(v1 : Vector, v2 : Vector) -> Double {
        var total : Double = 0

        for i in (0..<v1.size) {
            total += (v1[i] * v2[i])
        }

        return total
    }
    
    static func +(v1: Vector, v2: Vector) throws -> Vector{
        if v1.size != v2.size {
            throw VectorError.VEC_WRONG_SIZE
        }
        
        var result = Vector(v1.size)
        
        for i in (0..<result.size) {
            result[i] = v1[i] + v2[i]
        }
        
        return result
    }
    
    static func -(v1: Vector, v2: Vector) throws -> Vector{
        if v1.size != v2.size {
            throw VectorError.VEC_WRONG_SIZE
        }
        
        var result = Vector(v1.size)
        
        for i in (0..<result.size) {
            result[i] = v1[i] - v2[i]
        }
        
        return result
    }
    
    static func /(v1: Vector, v2: Vector) throws -> Vector{
        if v1.size != v2.size {
            throw VectorError.VEC_WRONG_SIZE
        }
        
        var result = Vector(v1.size)
        
        for i in (0..<result.size) {
            result[i] = v1[i] / v2[i]
        }
        
        return result
    }
    
    
    
    subscript(index : Int) -> Double {
        get {
            points[index]
        }
        set(value) {
            points[index] = value
        }
    }
}
