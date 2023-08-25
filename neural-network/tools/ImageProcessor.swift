//
//  ImageProcessor.swift
//  neural-network
//
//  Created by FVFH4069Q6L7 on 23/08/2023.
//

import AppKit
import Foundation

enum Channel : Int{
    case RED = 0
    case GREEN = 1
    case BLUE = 2
}

struct Pair {
    var x : Int
    var y : Int
    
    init(_ x: Int, _ y: Int) {
        self.x = x
        self.y = y
    }
    
    static func += (lhs : inout Pair, rhs : Pair) {
        lhs.x += rhs.x
        lhs.y += rhs.y
    }
}

struct RGB {
    var r : Double = 0
    var g : Double = 0
    var b: Double = 0
    
    init(_ randomise : Bool) {
        if randomise {
            r = Double.random(in: 0...255)
            g = Double.random(in: 0...255)
            b = Double.random(in: 0...255)
        }
    }
    
    init(_ color :NSColor) {
        r = color.redComponent * 255
        g = color.greenComponent * 255
        b = color.blueComponent * 255
    }
    
    func distFrom(target : RGB) -> Double {
        let rDiff = r - target.r
        let gDiff = g - target.g
        let bDiff = b - target.b
        
        return sqrt(rDiff * rDiff + gDiff * gDiff + bDiff * bDiff)
    }
    
    func toPixelArray() -> [Int] {
        var arr = Array.init(repeating: 255, count: 4)
        
        arr[0] = Int(r)
        arr[1] = Int(g)
        arr[2] = Int(b)
        
        return arr
    }
    
    mutating func reset () {
        r = 0
        g = 0
        b = 0
    }
    
    static func += (lhs : inout RGB, rhs : RGB) {
        lhs.r += rhs.r
        lhs.g += rhs.g
        lhs.b += rhs.b
    }
    
    static func /=(lhs : inout RGB, rhs : Double) {
        lhs.r /= rhs
        lhs.g /= rhs
        lhs.b /= rhs
    }
}

struct ImageProcessor {
    private var rep : NSBitmapImageRep?
    private static let ySobel = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]
    
    private static let xSobel = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]
    
    init(rep : NSBitmapImageRep) {
        self.rep = rep
    }
    
    init(_ fileURL : URL) throws {
        let imgData = try Data(contentsOf: fileURL)
        let imgOrig = NSImage(data: imgData)

        let rep = NSBitmapImageRep(bitmapDataPlanes: nil,
                                   pixelsWide: Int(imgOrig!.size.width),
                                   pixelsHigh: Int(imgOrig!.size.height),
                                   bitsPerSample: 8,
                                   samplesPerPixel: 4,
                                   hasAlpha: true,
                                   isPlanar: false,
                                   colorSpaceName: .deviceRGB,
                                   bytesPerRow: Int(imgOrig!.size.width) * 4,
                                   bitsPerPixel: 32)

        let ctx = NSGraphicsContext.init(bitmapImageRep: rep!)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep!)
        imgOrig!.draw(at: NSZeroPoint, from: NSZeroRect, operation: NSCompositingOperation.copy, fraction: 1.0)
        ctx?.flushGraphics()
        NSGraphicsContext.restoreGraphicsState()
        
        self.rep = rep
    }
    
    func hog4() -> Vector{
        let bins = createBins()
        
        var result = Vector(0)
        for i in stride(from: 0, to: bins.count - 1, by: 2) {
            for j in stride(from: 0, to: bins[0].count - 1, by: 2) {
                var thirtySixVec = Vector(0)
                thirtySixVec.concat(v: bins[i][j])
                thirtySixVec.concat(v: bins[i][j + 1])
                thirtySixVec.concat(v: bins[i + 1][j])
                thirtySixVec.concat(v: bins[i + 1][j + 1])
                
                thirtySixVec.normalize()
                result.concat(v: thirtySixVec)
            }
        }
        
        return result
    }
    
    internal func createBins() -> [[Vector]] {
        // applying sobel on all pixels
        let sobelResult = edgeDetecting()
        var bins : [[Vector]] = []
        
        var binCounts : Int = -1
        for x in stride(from: 0, to: sobelResult.count - 3, by: 4) {
            binCounts += 1
            bins.append([])
            for y in stride(from: 0, to: sobelResult[0].count - 3, by: 4) {
                var bin = Vector(10)
                
                // 4x4
                for i in (x..<(x + 4)) {
                    for j in (y..<(y + 4)) {
                        let binIndex = Int(sobelResult[i][j].1 / 20)
                        let leftBinValue = Double(binIndex) * 20.0
                        let rightBinValue = (Double(binIndex) + 1) * 20.0
                        
                        //More further away, less value gained
                        
                        let leftPercent = (rightBinValue - sobelResult[i][j].1) / (rightBinValue - leftBinValue)
                        
                        let rightPercent = (sobelResult[i][j].1 - leftBinValue) / (rightBinValue - leftBinValue)
                        
                        if binIndex != -1 {
                            bin[binIndex] += sobelResult[i][j].0 * leftPercent
                        }
                        
                        if abs(rightPercent) < 1e-7 {
                            bin[binIndex + 1] += sobelResult[i][j].0 * rightPercent
                        }
                    }
                }
                
                bins[binCounts].append(bin)
            }
        }
        
        return bins
    }
    
    internal func edgeDetecting() -> [[(Double, Double)]] {
        if rep == nil {
            return []
        }
        
        var gradientsAndAngles : [[(Double,  Double)]]  = []
        
        for y in 0..<(rep!.pixelsHigh) {
            gradientsAndAngles.append([])
            
            for x in 0..<(rep!.pixelsWide) {
                let xDiff = convolve3(
                    y: y,
                    x: x,
                    convolveOperator: ImageProcessor.xSobel)
                
                let yDiff = convolve3(
                    y: y,
                    x: x,
                    convolveOperator: ImageProcessor.ySobel)
                
                let gradientMagnitude = sqrt(Double(xDiff * xDiff + yDiff * yDiff))
                
                let angleInDegrees =
                    atan(Double(yDiff) / (Double(xDiff) + 1e-12)) * 180 / Double.pi + 90
                gradientsAndAngles[y].append((gradientMagnitude, angleInDegrees))
            }
        }
        
        return gradientsAndAngles
    }
    
    internal func convolve3(y : Int, x : Int, convolveOperator : [[Int]]) -> Int {
        var total : Int = 0

        for i in y...min(rep!.pixelsHigh - 1, (y + 2)) {
            for j in x...min(rep!.pixelsWide - 1, (x + 2)) {
                let intensity = Int(rep!.colorAt(x: j, y: i)!.redComponent * 255)
                
                total += intensity * Int(convolveOperator[j - x][i - y])
            }
        }
        
        return total
    }
}
