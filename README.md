# mlperp

Framework do trenowania modeli predykcji szeregów czasowych (shallow + deep).

## Struktura

- `klasy_data/` — przygotowanie danych (DataConfig, PdsSetup, KluskiConfig, KdsSetup)
- `klasy_jesien/` — architektury modeli (shallow: RFR/XGB/LR; deep: BiGRU+Attention, LSTM, Encoder-Decoder) oraz runner
- `klasy_tuner/` — strojenie hiperparametrów
- `inputs/` — konfiguracja kolumn i hiperparametrów modeli
- `adam*.py` — skrypty uruchomieniowe dla poszczególnych projektów (tg, mr, gda)

## Projekty

- **tg** — turbina parowa (przewidywanie temperatury obudowy)
- **mr** — maszyna rezerwy
- **gda** — EC Elbląg (zapotrzebowanie na ciepło i ceny energii/gazu)

## Architektury deep

| `dm_type` | Opis |
|-----------|------|
| `bigruta` | BiGRU + Attention → Dense(1), seq2one |
| `lstm` | LSTM → Dense(1), seq2one |
| `bigruta_seq2seq` | BiGRU + Attention → Dense(output_steps) |
| `enc_dec` | Encoder GRU → Decoder GRU → Dense, nieautoregresyjny |
