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
    "Play alt rock",
    "too loud",
    "Skip this",
    "Next one please",
    "Change song",
    "play Comfortably Numb",
    "Go to lst song",
    "What's the time?",
    "pause it",
    "make it quieter",
    "What should I eat?",
    "off",
    "Start focus music",
    "unmute",
    "What's your favorite color?",
]

let inferenceOptionsBuilder = InferenceOptionsBuilder()
try inferenceOptionsBuilder.withTemperature(temperature: 0.0)
try inferenceOptionsBuilder.withTokenCount(contextWindow: 50)
let inferenceOptions = try inferenceOptionsBuilder.build()

for prompt in prompts {
    let result = try model.runInference(promptText: prompt, conversationContext: context, inferenceOptions: inferenceOptions)
    print("\(prompt)   ->   \(result.resultText)")
}
