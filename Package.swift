// swift-tools-version: 6.0

import PackageDescription

let mlxDependencies: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"),
]

let cliDependencies: [Target.Dependency] = [
    "MLXPE",
    .product(name: "ArgumentParser", package: "swift-argument-parser"),
]

let strictConcurrency: [SwiftSetting] = [
    .enableExperimentalFeature("StrictConcurrency")
]

var package = Package(
    name: "mlx-swift-pe",
    platforms: [
        .macOS(.v14),
        .iOS(.v16),
    ],
    products: [
        .library(name: "MLXPE", targets: ["MLXPE"]),
        .executable(name: "pe-encode", targets: ["Encode"]),
        .executable(name: "pe-search", targets: ["Search"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift.git", .upToNextMinor(from: "0.30.6")),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "MLXPE",
            dependencies: mlxDependencies,
            path: "Sources/MLXPE",
            swiftSettings: strictConcurrency
        ),
        .executableTarget(
            name: "Encode",
            dependencies: cliDependencies,
            path: "Tools/encode",
            swiftSettings: strictConcurrency
        ),
        .executableTarget(
            name: "Search",
            dependencies: cliDependencies,
            path: "Tools/search",
            swiftSettings: strictConcurrency
        ),
        .testTarget(
            name: "MLXPETests",
            dependencies: ["MLXPE"],
            path: "Tests/MLXPETests",
            resources: [.copy("Resources")],
            swiftSettings: strictConcurrency
        ),
    ]
)

if Context.environment["MLX_SWIFT_BUILD_DOC"] == "1"
    || Context.environment["SPI_GENERATE_DOCS"] == "1"
{
    package.dependencies.append(
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.3.0")
    )
}
