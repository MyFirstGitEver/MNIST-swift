//
//  main.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import Foundation

//let reader = ExcelReader("/Users/FVFH4069Q6L7/Desktop/ML-data/framingham.xlsx")
//
//let(trainSet, testSet) = try reader.split(0, 0.7)
//
//let model = SimpleNeuralNetwork(
//    dataset: trainSet.map {
//        if $0.1 == 1.0 {
//            return ($0.0, Vector(data: [0.0, 1.0]))
//        }
//
//        return ($0.0, Vector(data: [1.0, 0.0]))
//    },
//    layers: [
//        Layer(inputSize: 15, outputSize: 4, activation: ReluActivation()),
//        Layer(inputSize: 4, outputSize: 2, activation: SoftmaxActivation()),
//    ], costFunction: CrossEntropy())
//
//print(try model.cost())
//try model.train(
//    batchSize: 80,
//    iteration: 50,
//    learningRate: 0.001, maxToSave: 5)
//
//var hit = 0.0
//for point in testSet {
//    let prediction = try model.forward(x: point.0)
//
//    if prediction[0] < prediction[1] && point.1 == 1.0 {
//        hit += 1.0
//    }
//    else if prediction[0] > prediction[1] && point.1 == 0.0 {
//        hit += 1.0
//    }
//}
//
//print(hit / Double(testSet.count) * 100)

//if let desktopURL = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first {
//    var dataset : [(Vector, Vector)] = []
//
//    for digit in (0...9) {
//        let url = desktopURL.appending(component: "ML-data/MNIST/training/\(digit)")
//        let filePaths =
//            try FileManager.default.contentsOfDirectory(atPath: url.path())
//
//        var pathCount = 0
//        for path in filePaths {
//            let hogVec = try ImageProcessor(url.appending(component: path)).hog4()
//            var label = Vector(10)
//
//            label[digit] = 1.0
//            dataset.append((hogVec, label))
//
//            pathCount += 1
//            if (pathCount % 500 == 0) {
//                print("Scanning digits of \(digit). \(pathCount)/\(filePaths.count)")
//            }
//        }
//    }
//
//    let model = SimpleNeuralNetwork(dataset: dataset, layers: [
//        Layer(inputSize: 360, outputSize: 10, activation: ReluActivation()),
//        Layer(inputSize: 10, outputSize: 10, activation: SoftmaxActivation())
//    ], costFunction: CrossEntropy())
//
//    try model.loadParams()
//
//    try model.train(batchSize: 200, iteration: 20, learningRate: 0.001, maxToSave: 1)
//}

let model = SimpleNeuralNetwork(dataset: [], layers: [
    Layer(inputSize: 360, outputSize: 10, activation: ReluActivation()),
    Layer(inputSize: 10, outputSize: 10, activation: SoftmaxActivation())
], costFunction: CrossEntropy())

try model.loadParams()

if let desktopURL = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first {
    for digit in (0...9) {
        let url = desktopURL.appending(component: "ML-data/MNIST/testing/\(digit)")
        let filePaths =
        try FileManager.default.contentsOfDirectory(atPath: url.path())

        var pathCount = 0
        var hit = 0

        for path in filePaths {
            let hogVec = try ImageProcessor(url.appending(component: path)).hog4()
            let predictionVec = try model.forward(x: hogVec)

            var prediction = 0
            var confidence = predictionVec[0]

            for i in (1..<predictionVec.size) {
                if confidence < predictionVec[i] {
                    confidence = predictionVec[i]
                    prediction = i
                }
            }

            if prediction == digit {
                hit += 1
            }

            pathCount += 1
            if (pathCount % 500 == 0) {
                print("Testing digits of \(digit). \(pathCount)/\(filePaths.count)")
            }
        }

        print("Accuracy: \(Double(hit) / Double(pathCount) * 100.0)%")
    }
}
