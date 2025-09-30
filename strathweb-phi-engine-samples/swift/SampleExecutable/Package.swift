// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SampleExecutable",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        .executableTarget(
            name: "SampleExecutable",
            dependencies: ["Strathweb.Phi.Engine.FFI"],
            path: "Sources/Executable",
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("SystemConfiguration"),
                .linkedLibrary("c++")
            ]
        ),
        .target(
            name: "Strathweb.Phi.Engine.FFI",
            dependencies: ["strathweb_phi_engine_framework"],
            path: "Sources/FFI",
            publicHeadersPath: "include"),
        .binaryTarget(
            name: "strathweb_phi_engine_framework",
            path: "Lib/strathweb_phi_engine_framework.xcframework"),
    ]
)
