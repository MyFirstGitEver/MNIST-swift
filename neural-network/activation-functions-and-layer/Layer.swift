//
//  Layer.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import Foundation

class Layer {
    private var W: Matrix
    private var b: Vector
    private var activation: ActivationFunction
    
    var inputSize : Int {
        return W.getShape().1
    }
    
    var outputSize : Int {
        return W.getShape().0
    }
    
    init(inputSize: Int, outputSize: Int, activation: ActivationFunction) {
        W = Matrix(firstDimension: outputSize, secondDimension: inputSize)
        b = Vector(outputSize)
        self.activation = activation
        
        b.randomise()
        W.randomise()
    }
    
    func fit(x: Vector) throws -> (Vector, Vector){
        return try (W * x + b, activation.f(z: try (W * x + b)))
    }
    
    func backward(input: Vector, index: Int) -> Vector {
        return activation.derivative(x: input, index: index)
    }
    
    func getTransposeOfW() -> Matrix {
        return W.transpose()
    }
    
    func update(
        firstMomentW: Matrix,
        secondMomenW: Matrix,
        firstMomentB: Vector,
        secondMomentB: Vector,
        batchSize: Int,
        learningRate: Double) throws {
            W = W - (firstMomentW / secondMomenW.sqrt()).scaled(factor: learningRate)
            
            b = try b - (firstMomentB / secondMomentB.sqrt()).scaled(factor: learningRate)
        }
    
    func loadW(data: Data) throws {
        self.W = try JSONDecoder().decode(Matrix.self, from: data)
    }
    
    func loadB(data: Data) throws {
        self.b = try JSONDecoder().decode(Vector.self, from: data)
    }
    
    func getWData() throws -> Data {
        return try JSONEncoder().encode(W)
    }

    func getBData() throws -> Data {
        return try JSONEncoder().encode(b)
    }
}
