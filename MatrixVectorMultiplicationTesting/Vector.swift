//
//  Vector.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import Foundation

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

    func identical(_ vecToCheck : Vector) -> Bool {
        if size != vecToCheck.size {
            return false
        }
        
        for i in (0..<(vecToCheck.size)) {
            if (abs(points[i] - vecToCheck[i]) > 1e-8) {
                return false
            }
        }
        
        return true
    }
    
    mutating func normalize() {
        var magnitude = 0.0
        
        for i in (0..<points.count) {
            magnitude += points[i] * points[i]
        }
        
        magnitude = magnitude.squareRoot()
        
        for i in (0..<points.count) {
            points[i] /= magnitude
        }
    }
    
    mutating func concat(v: Vector) {
        var newVectorData = Array.init(repeating: 0.0, count: size + v.size)
        
        for i in (0..<size) {
            newVectorData[i] = points[i]
        }
        
        for i in (size..<(v.size + size)) {
            newVectorData[i] = v[i - size]
        }
        
        points = newVectorData
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

    static func *(v1 : Vector, v2 : Vector) -> Double {
        var total : Double = 0

        for i in (0..<v1.size) {
            total += (v1[i] * v2[i])
        }

        return total
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
