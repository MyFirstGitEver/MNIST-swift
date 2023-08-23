//
//  ExcelReader.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 21/08/2023.
//

import Foundation
import CoreXLSX

struct ExcelReader {
    private let filePath : String
    
    init(_ filePath : String) {
        self.filePath = filePath
    }
    
    func split(
        _ labelColIndex : Int,
        _ trainPercent : Double) throws -> ([(Vector, Double)], [(Vector, Double)]) {
            var trainSet : [(Vector, Double)] = []
            var testSet : [(Vector, Double)] = []
            
            guard let file = XLSXFile(filepath: filePath)
            else {
              fatalError("XLSX file at \(filePath) is corrupted or does not exist")
            }

            for wbk in try file.parseWorkbooks() {
                for (_, path) in try file.parseWorksheetPathsAndNames(workbook: wbk) {
                    let worksheet = try file.parseWorksheet(at: path)
                    
                    let rows = worksheet.data?.rows
                    let colCount = rows![0].cells.count - 1
                    
                    var colsStringTypeIndexer = (0..<colCount).map { _ in [String:Int]() }
                    
                    let trainSize = Int(Double(rows!.count) * trainPercent)
                    
                    for (rowIndex, row) in (rows ?? []).enumerated() {
                        if rowIndex == 0 {
                            continue
                        }

                        var v = Vector(row.cells.count - 1) // account for label column
                        
                        var y = 0.0
                        var vIndex = 0
                        
                        for (index, c) in row.cells.enumerated() {
                            
                            if index != labelColIndex {
                                v[vIndex] = convertFieldToNumber(
                                    value: c.value!,
                                    index: index,
                                    colsStringTypeIndexer: &colsStringTypeIndexer)
            
                                vIndex += 1
                            }
                            else {
                                y = convertFieldToNumber(
                                    value: c.value!,
                                    index: index,
                                    colsStringTypeIndexer: &colsStringTypeIndexer)
                            }
                        }
                        
                        if rowIndex < trainSize {
                            trainSet.append((v, y))
                        }
                        else {
                            testSet.append((v, y))
                        }
                    }
                }
            }
            
            return (trainSet, testSet)
        }
    
    func convertFieldToNumber(
        value: String,
        index: Int,
        colsStringTypeIndexer: inout [[String: Int]]) -> Double {
        if let unknownTypeValue = Double(value) {
            return unknownTypeValue
        }
        else if(colsStringTypeIndexer[index].contains(where: {
            $0.key == value
        })) {
            return Double(colsStringTypeIndexer[index][value]!)
        }
        else {
            colsStringTypeIndexer[index][value] = colsStringTypeIndexer.count
            return Double(colsStringTypeIndexer.count - 1)
        }
    }
}
