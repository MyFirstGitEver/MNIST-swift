//
//  main.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import Foundation

enum NeuralNetworkError: Error {
    case CANT_FORWARD
}

class SimpleNeuralNetwork {
    private var dataset: [(Vector, Vector)]
    private var layers : [Layer]
    private var costFunction: CostFunction
    
    private let beta = 0.9;
    private let beta2 = 0.999;
    
    init(dataset: [(Vector, Vector)], layers: [Layer], costFunction : CostFunction) {
        self.dataset = dataset
        self.layers = layers
        self.costFunction = costFunction
    }
    
    func forward(x: Vector) throws -> Vector {
        if layers[0].inputSize != x.size {
            throw NeuralNetworkError.CANT_FORWARD
        }
        
        var resultOfForwarding = x
        
        for layer in layers {
            resultOfForwarding = try layer.fit(x: resultOfForwarding).1
        }
        
        return resultOfForwarding
    }
    
    func train(
        batchSize: Int,
        iteration: Int,
        learningRate: Double,
        maxToSave: Int) throws {
            var firstMomentW : [Matrix] = []
            var secondMomentW : [Matrix] = []
            firstMomentW = (0..<layers.count).map {
                Matrix(
                    firstDimension: layers[$0].outputSize,
                    secondDimension: layers[$0].inputSize)
            }
            secondMomentW = (0..<layers.count).map {
                Matrix(
                    firstDimension: layers[$0].outputSize,
                    secondDimension: layers[$0].inputSize)
            }
            
            var firstMomentB = (0..<layers.count).map {
                Vector(layers[$0].outputSize)
            }
            
            var secondMomentB = (0..<layers.count).map {
                Vector(layers[$0].outputSize)
            }
            
            for iter in (0..<iteration) {
                if(iter % maxToSave == 0) {
                    print("\(iter) iterations have passed. Cost: \(try cost())")
                }
                
                for dataPointIndex in stride(
                    from: 0,
                    to: dataset.count,
                    by: batchSize) {
                    let from = dataPointIndex
                    let to = min(dataset.count, dataPointIndex + batchSize) - 1
                    
                    var (gradientW, gradientB) = try computeGradient(
                        from: from,
                        to: to)
                    
                    for i in (0..<layers.count) {
                        gradientW[i] = gradientW[i].scaled(factor: 1.0 / Double((to - from + 1)))
                        gradientB[i] = gradientB[i].scaled(factor: 1.0 / Double((to - from + 1)))
                        
                        firstMomentW[i] = firstMomentW[i].scaled(factor: beta)
                        + gradientW[i].scaled(factor: 1 - beta)
                        
                        secondMomentW[i] = secondMomentW[i].scaled(factor: beta2)
                        + gradientW[i].squared().scaled(factor: 1 - beta2)
                        
                        firstMomentB[i] = try firstMomentB[i].scaled(factor: beta)
                        + gradientB[i].scaled(factor: 1 - beta)
                        
                        secondMomentB[i] = try secondMomentB[i].scaled(factor: beta2)
                        + gradientB[i].squared().scaled(factor: 1 - beta2)
                        
                        try layers[i].update(
                            firstMomentW: firstMomentW[i],
                            secondMomenW: secondMomentW[i],
                            firstMomentB: firstMomentB[i],
                            secondMomentB: secondMomentB[i],
                            batchSize: (to - from + 1),
                            learningRate: learningRate)
                    }
                }
            }
        }
    
    func cost() throws -> Double {
        var cost = 0.0
        
        for dataPoint in dataset {
            let output = try forward(x: dataPoint.0)
            cost += try costFunction.errorOnDataset(a: output, label: dataPoint.1)
        }
        
        return cost / Double(dataset.count)
    }
    
    internal func computeGradient(from: Int, to: Int) throws ->
    ([Matrix], [Vector]){
        var gradientW = (0..<layers.count).map {
            Matrix(
                firstDimension: layers[$0].outputSize,
                secondDimension: layers[$0].inputSize)
        }
        
        var gradientB = (0..<layers.count).map {
            Vector(layers[$0].outputSize)
        }
        
        
        for i in(from...to) {
            let point = dataset[i]
            
            var allZ = (0..<layers.count).map {
                Vector(layers[$0].outputSize)
            }
            var allA = (0..<layers.count + 1).map {
                if $0 == 0 {
                    return Vector(0)
                }
                
                return Vector(layers[$0 - 1].outputSize)
            }
            
            allA[0] = point.0
            
            // forward phase
            for i in 1...(layers.count) {
                let zAndA = try layers[i - 1].fit(x: allA[i - 1])
                allA[i] = zAndA.1
                allZ[i - 1] = zAndA.0
            }
            
            var currentError = error(
                z: allZ[allZ.count - 1],
                dA: try costFunction.derivativeAt(
                    a: allA[allA.count - 1],
                    label: point.1),
                layerIndex: layers.count - 1) // !!!
            
            
            for layerId in stride(from: layers.count - 1, to: -1, by: -1) {
                gradientW[layerId] = gradientW[layerId] + currentError.mulWithRowVector(v: allA[layerId])
                gradientB[layerId] = try
                gradientB[layerId] + currentError
                
                if layerId != 0 {
                    let dA = try layers[layerId].getTransposeOfW() * currentError
                    
                    currentError = error(
                        z: allZ[layerId - 1],
                        dA: dA,
                        layerIndex: layerId - 1)
                }
            }
        }
        
        return (gradientW, gradientB)
    }
    
    internal func error(z: Vector, dA: Vector, layerIndex: Int) -> Vector {
        var dz = Vector(z.size)
        
        for i in (0..<z.size) {
            do {
                dz[i] = try dA.hadamard(v: layers[layerIndex]
                    .backward(input: z, index: i)).sum()
            }
            catch _ {
                
            }
        }
        
        return dz
    }
    
    internal func loadInitialData() {
        layers[0].loadW(data: convertStringToW(layerId: 0))
        layers[0].loadB(data: convertStringToB(layerId: 0))
        
        layers[1].loadW(data: convertStringToW(layerId: 1))
        layers[1].loadB(data: convertStringToB(layerId: 1))
        
    }
    
    internal func convertStringToW(layerId: Int) -> [[Double]] {
        if let url = FileManager.default.urls(
            for: .desktopDirectory,
            in: .userDomainMask).first {
            let wPath = url.appending(path: "ML-data/layerstest 0/layer \(layerId + 1)/w")
            
            let rows = String(
                data: FileManager.default.contents(atPath: wPath.path(percentEncoded: false))!,
                encoding: .utf8)!.split(separator: "\n")
            
            var data : [[Double]] = []
            for row in rows {
                data.append(rowToArray(stringWithTabs: String(row)))
            }
            
            return data
        }
        
        return [[]]
    }
    
    internal func convertStringToB(layerId: Int) -> [Double] {
        if let url = FileManager.default.urls(
            for: .desktopDirectory,
            in: .userDomainMask).first {
            let bPath = url.appending(path: "ML-data/layerstest 0/layer \(layerId + 1)/b")
            
            let dataInStr = String(
                data: FileManager.default.contents(atPath: bPath.path(percentEncoded: false))!,
                encoding: .utf8)
            
            return rowToArray(stringWithTabs: dataInStr!)
        }
        
        return []
    }
    
    internal func rowToArray(stringWithTabs: String) -> [Double] {
        let data = stringWithTabs.split(separator: "\t")
        
        var dataArray : [Double] = []
        for point in data {
            dataArray.append(Double(point)!)
        }
        
        return dataArray
    }
}


let reader = ExcelReader("/Users/FVFH4069Q6L7/Desktop/ML-data/framingham.xlsx")

let(trainSet, testSet) = try reader.split(0, 0.7)

let model = SimpleNeuralNetwork(
    dataset: trainSet.map {
        if $0.1 == 1.0 {
            return ($0.0, Vector(data: [0.0, 1.0]))
        }
        
        return ($0.0, Vector(data: [1.0, 0.0]))
    },
    layers: [
        Layer(inputSize: 15, outputSize: 4, activation: ReluActivation()),
        Layer(inputSize: 4, outputSize: 2, activation: SoftmaxActivation()),
    ], costFunction: CrossEntropy())

try model.train(
    batchSize: 80,
    iteration: 50,
    learningRate: 0.001, maxToSave: 5)

var hit = 0.0
for point in testSet {
    let prediction = try model.forward(x: point.0)
    
    if prediction[0] < prediction[1] && point.1 == 1.0 {
        hit += 1.0
    }
    else if prediction[0] > prediction[1] && point.1 == 0.0 {
        hit += 1.0
    }
}

print(hit / Double(testSet.count) * 100)
