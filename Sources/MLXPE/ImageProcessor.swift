import CoreGraphics
import Foundation
import MLX

#if canImport(UIKit)
    import UIKit

    /// Platform image type for iOS and related Apple platforms.
    public typealias PlatformImage = UIImage
#endif

#if canImport(AppKit)
    import AppKit

    /// Platform image type for macOS.
    public typealias PlatformImage = NSImage
#endif

struct ImageNormalization: Sendable {
    let mean: [Float]
    let std: [Float]

    init(mean: [Float], std: [Float]) {
        precondition(mean.count == 3 && std.count == 3, "Expected RGB mean/std")
        self.mean = mean
        self.std = std
    }

    static let perceptionEncoderDefault = ImageNormalization(
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5]
    )
}

enum ImageProcessorError: Error {
    case invalidImage
    case pixelBufferUnavailable
}

final class ImageProcessor {
    let imageSize: Int
    let centerCrop: Bool
    let normalization: ImageNormalization

    init(
        imageSize: Int,
        centerCrop: Bool = false,
        normalization: ImageNormalization = .perceptionEncoderDefault
    ) {
        self.imageSize = imageSize
        self.centerCrop = centerCrop
        self.normalization = normalization
    }

    func process(image: PlatformImage) throws -> MLXArray {
        guard let cgImage = Self.cgImage(from: image) else {
            throw ImageProcessorError.invalidImage
        }

        let targetPixels = try Self.resizeAndExtractRGB(
            cgImage: cgImage,
            imageSize: imageSize,
            centerCrop: centerCrop
        )

        let normalized = Self.normalize(
            rgbPixels: targetPixels,
            mean: normalization.mean,
            std: normalization.std
        )

        return MLXArray(normalized, [1, imageSize, imageSize, 3])
    }

    func callAsFunction(_ image: PlatformImage) throws -> MLXArray {
        try process(image: image)
    }

    private static func normalize(rgbPixels: [UInt8], mean: [Float], std: [Float]) -> [Float] {
        var output = [Float](repeating: 0, count: rgbPixels.count)
        for i in 0 ..< rgbPixels.count {
            let channel = i % 3
            let value = Float(rgbPixels[i]) / 255.0
            output[i] = (value - mean[channel]) / std[channel]
        }
        return output
    }

    private static func resizeAndExtractRGB(cgImage: CGImage, imageSize: Int, centerCrop: Bool)
        throws -> [UInt8]
    {
        let sourceWidth = cgImage.width
        let sourceHeight = cgImage.height

        let drawWidth: Int
        let drawHeight: Int

        if centerCrop {
            let scale = Float(imageSize) / Float(min(sourceWidth, sourceHeight))
            drawWidth = Int(round(Float(sourceWidth) * scale))
            drawHeight = Int(round(Float(sourceHeight) * scale))
        } else {
            drawWidth = imageSize
            drawHeight = imageSize
        }

        guard let scaledContext = makeRGBContext(width: drawWidth, height: drawHeight) else {
            throw ImageProcessorError.pixelBufferUnavailable
        }
        scaledContext.interpolationQuality = .high
        scaledContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: drawWidth, height: drawHeight))

        guard let scaledBytes = scaledContext.data else {
            throw ImageProcessorError.pixelBufferUnavailable
        }
        let scaledBuffer = scaledBytes.bindMemory(
            to: UInt8.self, capacity: drawWidth * drawHeight * 4)

        let cropX: Int
        let cropY: Int
        if centerCrop {
            cropX = max(0, (drawWidth - imageSize) / 2)
            cropY = max(0, (drawHeight - imageSize) / 2)
        } else {
            cropX = 0
            cropY = 0
        }

        var output = [UInt8](repeating: 0, count: imageSize * imageSize * 3)
        for y in 0 ..< imageSize {
            for x in 0 ..< imageSize {
                let sx = x + cropX
                let sy = y + cropY
                let srcIndex = ((sy * drawWidth) + sx) * 4
                let dstIndex = ((y * imageSize) + x) * 3
                output[dstIndex + 0] = scaledBuffer[srcIndex + 0]
                output[dstIndex + 1] = scaledBuffer[srcIndex + 1]
                output[dstIndex + 2] = scaledBuffer[srcIndex + 2]
            }
        }

        return output
    }

    private static func makeRGBContext(width: Int, height: Int) -> CGContext? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        return CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
    }

    private static func cgImage(from image: PlatformImage) -> CGImage? {
        #if canImport(UIKit)
            return image.cgImage
        #else
            var rect = CGRect(origin: .zero, size: image.size)
            return image.cgImage(forProposedRect: &rect, context: nil, hints: nil)
        #endif
    }
}
