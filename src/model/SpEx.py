import torch
import torch.nn as nn

from model.module import LSTM, TCNBlock


class Model(nn.Module):
    def __init__(
            self,
            short_win_size=20,
            middle_win_size=80,
            long_window_size=160,
            speech_encoder_out_channels=256,
            speaker_encoder_n_layers=1,
            speaker_encoder_hidden_channels=256,
            speaker_encoder_out_channels=400,
            tcn_input_channels=256,
            tcn_hidden_channels=512,
            n_stacks=4,
            n_tcn_blocks=8
    ):
        super().__init__()
        stride = short_win_size // 2

        self.short_speech_encoder = nn.Sequential(
            nn.Conv1d(1, speech_encoder_out_channels, short_win_size, stride=stride, bias=False, padding=short_win_size // 2 - 1),
            nn.ReLU()
        )
        self.middle_speech_encoder = nn.Sequential(
            nn.Conv1d(1, speech_encoder_out_channels, middle_win_size, stride=stride, bias=False, padding=middle_win_size // 2 - 1),
            nn.ReLU()
        )
        self.long_speech_encoder = nn.Sequential(
            nn.Conv1d(1, speech_encoder_out_channels, long_window_size, stride=stride, bias=False, padding=long_window_size // 2 - 1),
            nn.ReLU()
        )

        self.speaker_encoder = nn.Sequential(
            LSTM(
                input_size=1, hidden_size=speaker_encoder_hidden_channels,
                num_layers=speaker_encoder_n_layers, batch_first=True,
                bidirectional=True
            ),
            nn.ReLU(),
            nn.Linear(speaker_encoder_hidden_channels * 2, speaker_encoder_out_channels)
        )

        self.speech_extraction_encoder = nn.Sequential(
            nn.GroupNorm(1, 3 * speech_encoder_out_channels, eps=1e-8),
            nn.Conv1d(3 * speech_encoder_out_channels, tcn_input_channels, kernel_size=1)
        )

        self.speech_extractor = nn.ModuleList([])
        for s in range(n_stacks):
            for t in range(n_tcn_blocks):
                if t == 0:
                    self.speech_extractor.append(
                        TCNBlock(
                            in_channels=(speaker_encoder_out_channels + tcn_input_channels),
                            hidden_channel=tcn_hidden_channels,
                            out_channels=tcn_input_channels,
                            dilation=2**t,
                            padding=2**t,
                            use_skip_connection=False
                    ))
                else:
                    self.speech_extractor.append(
                        TCNBlock(
                            in_channels=tcn_input_channels,
                            hidden_channel=tcn_hidden_channels,
                            out_channels=tcn_input_channels,
                            dilation=2**t,
                            padding=2**t,
                            use_skip_connection=True
                    ))

        self.short_speech_extraction_decoder = nn.Sequential(
            nn.Conv1d(tcn_input_channels, speech_encoder_out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.middle_speech_extraction_decoder = nn.Sequential(
            nn.Conv1d(tcn_input_channels, speech_encoder_out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.long_speech_extraction_decoder = nn.Sequential(
            nn.Conv1d(tcn_input_channels, speech_encoder_out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.short_speech_decoder = nn.ConvTranspose1d(speech_encoder_out_channels, 1, short_win_size, stride=stride, bias=False, padding=short_win_size // 2 - 5)
        self.middle_speech_decoder = nn.ConvTranspose1d(speech_encoder_out_channels, 1, middle_win_size, stride=stride, bias=False, padding=middle_win_size // 2 - 5)
        self.long_speech_decoder = nn.ConvTranspose1d(speech_encoder_out_channels, 1, long_window_size, stride=stride, bias=False, padding=long_window_size // 2 - 5)

        self.n_tcn_blocks = n_tcn_blocks

    def forward(self, mixture, reference):
        """
            mixture: [batch_size, 1, 32000]
            reference: [batch_size, 1, T]
            short_scale_speech: [batch_size 1, 32000]
            middle_scale_speech:
            long_scale_speech:
            speaker_embedding_cache: [batch_size, 400]
        """
        short_scale_embedding = self.short_speech_encoder(mixture)  # [n_filters, 3200]
        middle_scale_embedding = self.middle_speech_encoder(mixture)  # [..., 3200]
        long_scale_embedding = self.long_speech_encoder(mixture)
        speech_embedding = torch.cat([short_scale_embedding, middle_scale_embedding, long_scale_embedding], dim=1)
        speech_embedding = self.speech_extraction_encoder(speech_embedding)  # [tcn_input_channels, 3200]
        # print("Speech embedding: ", speech_embedding.shape)

        speaker_embedding = self.speaker_encoder(reference).permute(0, 2, 1)  # [400, T]
        speaker_embedding = torch.mean(speaker_embedding, dim=-1)  # [400]
        speaker_embedding_cache = speaker_embedding
        speaker_embedding = speaker_embedding.unsqueeze(1).repeat((1, speech_embedding.shape[-1], 1)).permute(0, 2, 1)  # [400, 3200]
        # print("Speaker embedding: ", speaker_embedding.shape)

        for i in range(len(self.speech_extractor)):
            if (i % self.n_tcn_blocks) == 0:
                speech_embedding = torch.cat([speech_embedding, speaker_embedding], dim=1)  # [tcn_input_channels + 400 , 3200]
                speech_embedding = self.speech_extractor[i](speech_embedding)
            else:
                speech_embedding = self.speech_extractor[i](speech_embedding)

        short_scale_mask = self.short_speech_extraction_decoder(speech_embedding)
        middle_scale_mask = self.middle_speech_extraction_decoder(speech_embedding)
        long_scale_mask = self.long_speech_extraction_decoder(speech_embedding)

        short_scale_speech = self.short_speech_decoder(short_scale_mask * short_scale_embedding)
        middle_scale_speech = self.middle_speech_decoder(middle_scale_mask * middle_scale_embedding)
        long_scale_speech = self.long_speech_decoder(long_scale_mask * long_scale_embedding)

        return short_scale_speech, middle_scale_speech, long_scale_speech, speaker_embedding_cache

def test_32000():
    model = Model()
    mix = torch.rand(2, 1, 32000)
    ref = torch.rand(2, 1, 16384)
    model(mix, ref)


if __name__ == '__main__':
    test_32000()
