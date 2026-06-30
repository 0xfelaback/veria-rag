import asyncio
import time
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.logger import FrameLogger
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.frames.frames import (
    BotSpeakingFrame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    OutputAudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    AudioRawFrame,
    UserSpeakingFrame,
    SpeechControlParamsFrame,
    BotStartedSpeakingFrame,
    UserStartedSpeakingFrame,
    InterruptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import BotStoppedSpeakingFrame
import datetime
from loguru import logger as loggerr
from src.Infrastructure.llm.index import llm_pipecat
from src.Infrastructure.llm.query_function_call import (
    query_function_call,
    prewarm_embedding_model,
)

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/{current_time}-log-pipecat.log"
loggerr.add(
    log_filename,
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
)


prompt_text = "You are a charming and brilliant blockchain solutions architect speaking over a voice call. Maintain a warm, playfully flirty, yet deeply professional tone—just like a supportive, high-performing coworker. Keep every response naturally conversational, strictly limited to 1 or 2 short sentences, and always end with a clear cue or question to pass the mic back to the user."
min_prompt_text = "You are a helpful assistant."


class LatencyBenchmarkProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.start_time = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and "Pre-warm" in frame.text:
            self.start_time = time.perf_counter()
            print("\n [WARMUP] Synthesizing pre-warm text token array...")
        elif self.start_time and isinstance(frame, AudioRawFrame):
            elapsed = (time.perf_counter() - self.start_time) * 1000
            print(f" [BENCHMARK] Time-to-First-Audio (TTFA): {elapsed:.2f}ms")
            self.start_time = None

        # push the frame down the pipeline instead of returning it
        await self.push_frame(frame, direction)


class InterruptionCooldownProcessor(FrameProcessor):
    """Blocks mic input while the bot is speaking and adds a brief cooldown after."""

    def __init__(self, post_speech_cooldown=0.8):
        super().__init__()
        self.post_speech_cooldown = post_speech_cooldown
        self.bot_is_speaking = False
        self.bot_stop_time = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # track when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            self.bot_is_speaking = True
            print("[STATE] Bot is speaking. Mic input muted.")
            await self.push_frame(frame, direction)
            return

        # track when bot stops speaking and start the cooldown timer
        if isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_is_speaking = False
            self.bot_stop_time = time.perf_counter()
            print(
                f"[STATE] Bot stopped. Cooldown active for {self.post_speech_cooldown}s"
            )
            await self.push_frame(frame, direction)
            return

        # calculate if we are in the post-speech cooldown window
        in_cooldown_window = False
        if not self.bot_is_speaking and self.bot_stop_time:
            if (time.perf_counter() - self.bot_stop_time) < self.post_speech_cooldown:
                in_cooldown_window = True
            else:
                self.bot_stop_time = None

        if (self.bot_is_speaking or in_cooldown_window) and isinstance(
            frame,
            (
                UserStartedSpeakingFrame,
                InterruptionFrame,
                VADUserStartedSpeakingFrame,
                InputAudioRawFrame,
            ),
        ):
            print("[AEC BLOCK] Echo or interruption blocked.")
            return

        await self.push_frame(frame, direction)


async def main():
    vad_analyzer = SileroVADAnalyzer()

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = OpenAISTTService(
        api_key="not-needed",
        base_url="http://127.0.0.1:8080/v1",
        settings=OpenAISTTService.Settings(
            model="whisper-1",
            prompt="This is a conversation with Vicky, an automation engineer at Sterling Bank.",
        ),
    )

    llm = OLLamaLLMService(
        base_url="http://localhost:11434/v1",
        settings=OLLamaLLMService.Settings(model="llama3.2:3b"),
    )

    tts = OpenAITTSService(
        api_key="not-needed",
        base_url="http://localhost:8880/v1",
        settings=OpenAITTSService.Settings(
            # model="kokoro",
            model="gpt-4o-mini-tts",
            voice="ash",
        ),
    )

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": min_prompt_text,
            }
        ],
        tools=[query_function_call],
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context=context,
    )

    logger = FrameLogger(
        prefix="Pipeline Output Monitor",
        ignored_frame_types=(
            BotSpeakingFrame,
            UserSpeakingFrame,
            InputAudioRawFrame,
            OutputAudioRawFrame,
            TextFrame,
            LLMFullResponseEndFrame,
        ),
    )

    # Create interruption cooldown processor
    interruption_cooldown = InterruptionCooldownProcessor()

    pipeline = Pipeline(
        [
            transport.input(),
            interruption_cooldown,
            VADProcessor(vad_analyzer=vad_analyzer),
            stt,
            logger,
            user_aggregator,
            llm_pipecat,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    vad_params = SpeechControlParamsFrame(
        vad_params=VADParams(
            confidence=0.6,
            start_secs=0.2,
            min_volume=0.6,  # stop_secs=2.0,
            stop_secs=0.8,
        )
    )

    runner = WorkerRunner()
    worker = PipelineWorker(pipeline)
    await runner.add_workers(worker)

    print("\n [INFRA] Initializing complete local AI loop...")
    asyncio.create_task(runner.run())

    await asyncio.sleep(3)

    print("[INFRA] Pre-warming embedding model...")
    await prewarm_embedding_model()

    print("[INFRA] Pushing VAD configuration frame...")
    await pipeline.queue_frame(vad_params)

    print("[INFRA] Pushing pre-warm frame to system pipeline cache...")
    await pipeline.queue_frame(
        TranscriptionFrame(
            text="Pre-warm initialization test.", user_id="system", timestamp="now"
        )
    )

    print(
        "\n🎙️  [LIVE WITH VAD] Full System hot! Start speaking into your MacBook mic..."
    )

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
