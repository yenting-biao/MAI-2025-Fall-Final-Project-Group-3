import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModel,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)
import os
import librosa
from funasr import AutoModel


class AudioCaptioner:

    def __init__(
        self,
        whisper_model_id: str = "openai/whisper-large-v3",
        emotion_model_id: str = "iic/emotion2vec_plus_large",
        gender_model_id: str = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
    ):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad"
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils

        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_id)
        self.processor = WhisperProcessor.from_pretrained(whisper_model_id)

        self.emotion_model = AutoModel(model=emotion_model_id)

        self.genderlabel2id = {"female": 0, "male": 1}
        self.genderid2label = {0: "female", 1: "male"}
        self.gender_feature_extractor = AutoFeatureExtractor.from_pretrained(
            gender_model_id
        )
        self.gender_model = AutoModelForAudioClassification.from_pretrained(
            gender_model_id,
            num_labels=2,
            label2id=self.genderlabel2id,
            id2label=self.genderid2label,
        )

        # move gender model to device as well
        self.gender_model.to(self.device)
        self.whisper.to(self.device)

    def _format_seconds_to_hms(self, seconds_float: float) -> str:
        """
        Convert seconds (float) to HH:MM:SS format.
        """
        total_seconds = int(round(seconds_float))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def _capitalize_first_letter(self, text: str) -> str:
        if not text:
            return text
        return text[0].upper() + text[1:].lower()

    def _do_vad(self, audio_path: str):
        wav = self.read_audio(audio_path)
        speech_timestamps = self.get_speech_timestamps(
            wav, self.vad_model, return_seconds=True
        )
        return speech_timestamps

    def _do_asr(self, audio_path: str) -> str:
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        generated_ids = self.whisper.generate(input_features)
        transcription = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        return transcription

    def _do_SER(self, audio_path: str) -> dict:
        res = self.emotion_model.generate(
            audio_path,
            # output_dir="./outputs",
            granularity="utterance",
            extract_embedding=False,
        )[0]
        max_score_ind = res["scores"].index(max(res["scores"]))
        return res["labels"][max_score_ind].split("/")[-1]

    def _do_GR(self, audio_path: str) -> dict:
        sr = getattr(self.gender_feature_extractor, "sampling_rate", 16000)
        audio, _ = librosa.load(audio_path, sr=sr)

        inputs = self.gender_feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.gender_model.eval()
        with torch.no_grad():
            outputs = self.gender_model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = (
            int(torch.argmax(probs, dim=-1).cpu().numpy()[0])
            if probs.dim() > 1
            else int(torch.argmax(probs).cpu().numpy())
        )

        id2label = getattr(self.gender_model.config, "id2label", self.genderid2label)
        label = id2label.get(pred_id, self.genderid2label.get(pred_id, str(pred_id)))
        score = (
            float(probs[0, pred_id].cpu().item())
            if probs.dim() > 1
            else float(probs[pred_id].cpu().item())
        )

        prob_map = {}
        for i, p in enumerate(probs[0] if probs.dim() > 1 else probs):
            lbl = id2label.get(i, self.genderid2label.get(i, str(i)))
            prob_map[str(lbl)] = float(p.cpu().item())

        # return {"label": label, "score": score, "probabilities": prob_map}
        return label

    def get_caption_from_audio(self, audio_path: str) -> str:
        vad_segments = self._do_vad(audio_path)
        captions = []
        for segment in vad_segments:
            start_time = self._format_seconds_to_hms(segment["start"])
            end_time = self._format_seconds_to_hms(segment["end"])
            temp_audio_path = "temp_segment.wav"
            os.system(
                f"ffmpeg -y -i {audio_path} -ss {start_time} -to {end_time} -ar 16000 -ac 1 {temp_audio_path} > /dev/null 2>&1"
            )
            asr = self._do_asr(temp_audio_path)
            emotion = self._do_SER(temp_audio_path)
            emotion = self._capitalize_first_letter(emotion)
            gender = self._do_GR(temp_audio_path)
            gender = self._capitalize_first_letter(gender)
            captions.append(
                f"[{start_time} - {end_time}] {asr} (Gender: {gender}, Emotion: {emotion})"
            )
            os.remove(temp_audio_path)
        return "\n".join(captions)


if __name__ == "__main__":
    captioner = AudioCaptioner()
    audio_path = (
        "/Users/biao/files/NTU/碩班/碩一上/course/MAI/Final-Project/1001_DFA_ANG_XX.wav"
    )
    caption = captioner.get_caption_from_audio(audio_path)
    print(caption)
