import Foundation

let sourceFileDir = (#file as NSString).deletingLastPathComponent
let basePath = ((sourceFileDir as NSString).appendingPathComponent("../../../../../fine-tuning/fused_model") as NSString).standardizingPath

let modelBuilder = PhiEngineBuilder()
_ = try modelBuilder.tryUseGpu()
try modelBuilder.withModelProvider(modelProvider: PhiModelProvider.fileSystem(
    indexPath: "\(basePath)/model.safetensors.index.json",
    configPath: "\(basePath)/config.json"
))

let model = try modelBuilder.build(cacheDir: (sourceFileDir as NSString).appending("/.cache"))

let context = ConversationContext(messages: [], systemInstruction: "")
let prompts = [
    "Boost the volume",
    "What's your name again?",
    "Turn up",
    "Lower it",
    "Last song please",
    "Skip",
    "Stop all",
    "Audio enable",
    "Play Wish You Were Here"
]

let inferenceOptionsBuilder = InferenceOptionsBuilder()
try inferenceOptionsBuilder.withTemperature(temperature: 0.0)
try inferenceOptionsBuilder.withTokenCount(contextWindow: 50)
let inferenceOptions = try inferenceOptionsBuilder.build()

for prompt in prompts {
    let result = try model.runInference(promptText: prompt, conversationContext: context, inferenceOptions: inferenceOptions)
    print("\(prompt)   ->   \(result.resultText)")
}
