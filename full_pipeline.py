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
    UserStoppedSpeakingFrame,
    UserTurnInferenceCompletedFrame,
    VADUserStartedSpeakingFrame,
    FunctionCallsStartedFrame,
    FunctionCallResultFrame,
    FunctionCallCancelFrame,
    VADUserStoppedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMAssistantAggregatorParams,
)
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
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


prompt_text = (
    "You are Vicky, a blockchain architect. Be brief and direct. Answer in one "
    "short spoken sentence (<=15 words). NEVER use markdown, headings, or bullets. "
    "Answer directly from the latest retrieved context and preserve the technical "
    "meaning. Do not oversimplify or paraphrase away the core mechanism. If the "
    "answer is not in the context, say 'Context is insufficient.'"
)

min_prompt_text = (
    "You are a voice assistant. Your output must be spoken out loud. NEVER use "
    "markdown headings or bullet points. "
    "CRITICAL ROUTING RULE: If the user is greeting you, asking how you are, "
    "making small talk, or asking generic pleasantries, answer them directly "
    "in a single conversational sentence and STOP. Do NOT call any tools for small talk. "
    "FOR KNOWLEDGE QUERIES: You MUST call the query_function_call tool first to "
    "retrieve context. Limit your final answer to a single, direct sentence under "
    "15 words based strictly on the latest retrieved context. Preserve the technical "
    "meaning and key mechanism. Do not oversimplify. If the answer is not present "
    "in the context, say 'Context is insufficient.'"
)


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


class PipelineTimingProcessor(FrameProcessor):
    """Comprehensive timing processor to measure pipeline latency metrics."""

    def __init__(self):
        super().__init__()
        self.vad_stop_time = None
        self.user_stop_detected_time = None

        self.vad_stop_to_transcription_start = None
        self.transcription_complete_time = None

        self.function_call_start_time = None
        self.function_call_complete_time = None

        self.llm_start_time = None
        self.llm_first_token_time = None
        self.llm_complete_time = None
        self.llm_text_frame_count = 0

        self.tts_start_time = None
        self.tts_first_audio_time = None
        self.bot_started_speaking_time = None

        self.user_stop_to_bot_start_logged = False
        self.user_stop_to_bot_stop_logged = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        current_time = time.perf_counter()

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self.vad_stop_time = current_time
            loggerr.debug("[TIMING] VAD detected user stopped speaking")

        if isinstance(frame, UserStoppedSpeakingFrame):
            self.user_stop_detected_time = current_time
            self.user_stop_to_bot_start_logged = False
            self.user_stop_to_bot_stop_logged = False
            if self.vad_stop_time:
                vad_to_detection = (
                    self.user_stop_detected_time - self.vad_stop_time
                ) * 1000
                loggerr.info(
                    f"[TIMING] VAD stop to UserStoppedSpeaking: {vad_to_detection:.2f}ms"
                )
                self.vad_stop_time = None

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self.vad_stop_to_transcription_start = current_time

        if isinstance(frame, TranscriptionFrame):
            if getattr(frame, "final", False):
                self.transcription_complete_time = current_time
                if self.vad_stop_to_transcription_start:
                    vad_to_transcription = (
                        self.transcription_complete_time
                        - self.vad_stop_to_transcription_start
                    ) * 1000
                    loggerr.info(
                        f"[TIMING] VAD stop to Transcription complete: {vad_to_transcription:.2f}ms"
                    )
                    self.vad_stop_to_transcription_start = None

        if isinstance(frame, FunctionCallsStartedFrame):
            self.function_call_start_time = current_time
            loggerr.debug("[TIMING] Function call started")

        if isinstance(frame, FunctionCallResultFrame):
            self.function_call_complete_time = current_time
            if self.function_call_start_time:
                rag_time = (
                    self.function_call_complete_time - self.function_call_start_time
                ) * 1000
                loggerr.info(f"[TIMING] Function call execution time: {rag_time:.2f}ms")
                self.function_call_start_time = None

        if isinstance(frame, FunctionCallResultFrame):
            self.llm_start_time = current_time
            self.llm_text_frame_count = 0
            loggerr.debug("[TIMING] LLM generation started")

        if isinstance(frame, TextFrame) and hasattr(frame, "text") and frame.text:
            self.llm_text_frame_count += 1
            # Capture TTFT on first text frame
            if self.llm_start_time and not self.llm_first_token_time:
                self.llm_first_token_time = current_time
                ttft = (self.llm_first_token_time - self.llm_start_time) * 1000
                loggerr.info(f"[TIMING] LLM Time to First Token (TTFT): {ttft:.2f}ms")

        if isinstance(frame, LLMFullResponseEndFrame):
            self.llm_complete_time = current_time
            if self.llm_start_time:
                llm_generation_time = (
                    self.llm_complete_time - self.llm_start_time
                ) * 1000
                loggerr.info(
                    f"[TIMING] LLM total generation time: {llm_generation_time:.2f}ms"
                )
                if self.llm_first_token_time:
                    generation_after_first = (
                        self.llm_complete_time - self.llm_first_token_time
                    ) * 1000
                    loggerr.info(
                        f"[TIMING] LLM generation after TTFT: {generation_after_first:.2f}ms"
                    )
                loggerr.info(
                    f"[TIMING] LLM text frames generated: {self.llm_text_frame_count}"
                )
            # Reset LLM timing state
            self.llm_start_time = None
            self.llm_first_token_time = None
            self.llm_complete_time = None
            self.llm_text_frame_count = 0

        # 5. TTS timing (text to audio)
        if isinstance(frame, BotStartedSpeakingFrame):
            self.bot_started_speaking_time = current_time
            if self.llm_first_token_time:
                text_to_audio = (
                    self.bot_started_speaking_time - self.llm_first_token_time
                ) * 1000
                loggerr.info(
                    f"[TIMING] LLM first token to bot speaking: {text_to_audio:.2f}ms"
                )
            if self.transcription_complete_time:
                end_to_end = (
                    self.bot_started_speaking_time - self.transcription_complete_time
                ) * 1000
                loggerr.info(
                    f"[TIMING] End-to-end latency (transcription to audio): {end_to_end:.2f}ms"
                )
            if self.user_stop_detected_time and not self.user_stop_to_bot_start_logged:
                user_stop_to_bot_start = (
                    self.bot_started_speaking_time - self.user_stop_detected_time
                ) * 1000
                loggerr.info(
                    f"[TIMING] UserStoppedSpeaking to BotStartedSpeaking: {user_stop_to_bot_start:.2f}ms"
                )
                self.user_stop_to_bot_start_logged = True

        if isinstance(frame, BotStoppedSpeakingFrame):
            if self.bot_started_speaking_time:
                audio_duration = (current_time - self.bot_started_speaking_time) * 1000
                loggerr.info(
                    f"[TIMING] Audio playback duration: {audio_duration:.2f}ms"
                )
                self.bot_started_speaking_time = None
            if self.user_stop_detected_time:
                user_stop_to_bot_stop = (
                    current_time - self.user_stop_detected_time
                ) * 1000
                loggerr.info(
                    f"[TIMING] UserStoppedSpeaking to BotStoppedSpeaking: {user_stop_to_bot_stop:.2f}ms"
                )

        await self.push_frame(frame, direction)


class LLMContextPruningProcessor(FrameProcessor):
    """Trim old LLM context messages after each assistant turn."""

    def __init__(self, context: LLMContext, max_recent_turns: int = 2):
        super().__init__()
        self.context = context
        self.max_recent_turns = max_recent_turns

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStoppedSpeakingFrame):
            messages = self.context.messages
            if len(messages) > 8:

                def message_role(message):
                    if isinstance(message, dict):
                        return message.get("role")
                    return getattr(message, "role", None)

                system_messages = [
                    msg for msg in messages if message_role(msg) == "system"
                ]
                non_system_messages = [
                    msg for msg in messages if message_role(msg) != "system"
                ]

                kept_messages = []
                user_count = 0
                assistant_count = 0
                for msg in reversed(non_system_messages):
                    kept_messages.insert(0, msg)
                    role = message_role(msg)
                    if role == "user":
                        user_count += 1
                    elif role == "assistant":
                        assistant_count += 1
                    if (
                        user_count >= self.max_recent_turns
                        and assistant_count >= self.max_recent_turns
                    ):
                        break

                # Keep only the original system prompt plus the latest RAG system message.
                retained_system_messages = []
                if system_messages:
                    retained_system_messages.append(system_messages[0])
                    if len(system_messages) > 1:
                        retained_system_messages.append(system_messages[-1])

                pruned_messages = retained_system_messages + kept_messages
                self.context.set_messages(pruned_messages)
                loggerr.debug(
                    f"[CONTEXT] Pruned LLM history to last {self.max_recent_turns} turns and dropped older system messages."
                )

        await self.push_frame(frame, direction)


class ContextResetProcessor(FrameProcessor):
    """Clears stale user/tool context when a new explicit user turn begins."""

    def __init__(self, context: LLMContext):
        super().__init__()
        self.context = context

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(
            frame,
            (
                UserStartedSpeakingFrame,
                VADUserStartedSpeakingFrame,
                InterruptionFrame,
            ),
        ):
            messages = self.context.messages

            def message_role(message):
                if isinstance(message, dict):
                    return message.get("role")
                return getattr(message, "role", None)

            system_messages = [msg for msg in messages if message_role(msg) == "system"]
            if system_messages:
                self.context.set_messages([system_messages[0]])
                loggerr.debug(
                    "[CONTEXT] Cleared stale conversation history on new user turn."
                )

        await self.push_frame(frame, direction)


class InterruptionCooldownProcessor(FrameProcessor):
    """Blocks mic input while the bot is speaking, during function calls, and adds a brief cooldown after."""

    def __init__(self, post_speech_cooldown=0.8):
        super().__init__()
        self.post_speech_cooldown = post_speech_cooldown
        self.bot_is_speaking = False
        self.bot_stop_time = None
        self.function_call_in_progress = False

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

        if isinstance(frame, FunctionCallsStartedFrame):
            self.function_call_in_progress = True
            print("[STATE] Function call in progress. Mic input muted.")
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (FunctionCallResultFrame, FunctionCallCancelFrame)):
            self.function_call_in_progress = False
            print("[STATE] Function call completed. Mic input enabled.")
            await self.push_frame(frame, direction)
            return

        # calculate if we are in the post-speech cooldown window
        in_cooldown_window = False
        if not self.bot_is_speaking and self.bot_stop_time:
            if (time.perf_counter() - self.bot_stop_time) < self.post_speech_cooldown:
                in_cooldown_window = True
            else:
                self.bot_stop_time = None

        if (
            self.bot_is_speaking or in_cooldown_window or self.function_call_in_progress
        ) and isinstance(
            frame,
            (
                UserStartedSpeakingFrame,
                InterruptionFrame,
                VADUserStartedSpeakingFrame,
                InputAudioRawFrame,
            ),
        ):
            block_reason = (
                "bot speaking"
                if self.bot_is_speaking
                else "cooldown" if in_cooldown_window else "function call"
            )
            print(f"[AEC BLOCK] Echo or interruption blocked ({block_reason}).")
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
            prompt="This is a conversation with a blockchain wizard on technical blockchain technology questions.",
        ),
    )
    """llm = OLLamaLLMService(
        base_url="http://localhost:11434/v1",
        settings=OLLamaLLMService.Settings(
            model="llama3.2:3b",
            max_tokens=256,
        ),
        extra_body={"options": {"num_ctx": 2048}},
    )"""

    tts = OpenAITTSService(
        api_key="not-needed",
        base_url="http://localhost:8880/v1",
        settings=OpenAITTSService.Settings(
            model="kokoro",
            # model="gpt-4o-mini-tts",
            voice="alloy",
            speed=1,
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

    assistant_params = LLMAssistantAggregatorParams(
        enable_auto_context_summarization=False,
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context=context,
        assistant_params=assistant_params,
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
            # temporary frames
            # FunctionCallsStartedFrame,
            # UserTurnInferenceCompletedFrame,
            # VADUserStartedSpeakingFrame,
        ),
    )

    # Create interruption cooldown processor
    interruption_cooldown = InterruptionCooldownProcessor()

    timing_processor = PipelineTimingProcessor()
    pruning_processor = LLMContextPruningProcessor(context=context, max_recent_turns=2)
    context_reset_processor = ContextResetProcessor(context=context)

    pipeline = Pipeline(
        [
            transport.input(),
            interruption_cooldown,
            VADProcessor(vad_analyzer=vad_analyzer),
            stt,
            logger,
            timing_processor,
            user_aggregator,
            llm_pipecat,
            tts,
            transport.output(),
            assistant_aggregator,
            pruning_processor,
            context_reset_processor,
        ]
    )
    vad_params = SpeechControlParamsFrame(
        vad_params=VADParams(
            confidence=0.5,
            # confidence=0.8,  # for noisy environments
            start_secs=0.1,
            min_volume=0.5,
            stop_secs=1.8,
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

    print("[INFRA] Pre-warming LLM with test prompt...")
    await pipeline.queue_frame(
        TranscriptionFrame(text="Hello", user_id="system", timestamp="now")
    )
    await asyncio.sleep(2)

    print("[INFRA] Pushing VAD configuration frame...")
    await pipeline.queue_frame(vad_params)

    print("[INFRA] Pushing pre-warm frame to system pipeline cache...")
    await pipeline.queue_frame(
        TranscriptionFrame(
            # text="Pre-warm initialization test.",
            text="Hi, how are doing today.",
            user_id="system",
            timestamp="now",
        )
    )

    print(
        "\n🎙️  [LIVE WITH VAD] Full System hot! Start speaking into your MacBook mic..."
    )

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
