# FinPredict - Platforma za Analizu i Predikciju Kretanja Cijena Dionica

FinPredict je napredna, produkcijski spremna web platforma i sustav strojnog učenja razvijen za klasifikaciju i predikciju smjera kretanja cijena dionica (rast ili pad) za sljedeći trgovački dan (T+1). Projekt je razvijen kao cjelovito rješenje koje povezuje akademska istraživanja u području financijskih vremenskih serija s praktičnim, interaktivnim web sučeljem za vizualizaciju rezultata.

Sustav obuhvaća obradu povijesnih financijskih podataka za **54 dionice iz 6 različitih sektora**, treniranje i evaluaciju četiri napredna algoritma strojnog učenja te integraciju rezultata kroz klijent-poslužiteljsku web aplikaciju.

---

## 🚀 Ključne Značajke

- **Napredno inženjerstvo značajki (Feature Engineering):** Integracija 25 različitih tehničkih pokazatelja, uključujući RSI, MACD, Bollinger Bands, ATR, momentum, povijesne prinose, pomične prosjeke te makroekonomske indikatore (S&P 500 prinos i VIX indeks volatilnosti).
- **Hibridni strojni modeli:** Implementacija i usporedba četiri modela:
  - **LSTM (Long Short-Term Memory):** Duboki sekvencijalni model s pogledom unazad od 10 dana, optimiziran regularizacijom i Dropoutom.
  - **XGBoost (Extreme Gradient Boosting):** Stabilni stablo-bazirani model za nelinearne zavisnosti.
  - **Random Forest:** Ansambl metoda s kontroliranom dubinom stabla radi sprječavanja prenaučenosti.
  - **Logistička regresija:** Linearni model s L2 regularizacijom i skaliranjem za stabilne bazne procjene.
- **Optimizacija klasifikacijskog praga:** Automatizirano podešavanje praga na validacijskom skupu pomoću **Macro F1-Score** metrike radi ublažavanja utjecaja neuravnoteženosti klasa.
- **Aktivna simulacija trgovanja:** Integrirani modul za simulaciju u 2024. godini koji uspoređuje aktivno upravljanje modelom s pasivnom strategijom "Kupi i drži" (Buy & Hold), prateći povrat, alfu i završni kapital.
- **Premium Bloomberg-style UI:** Moderni klijentski panel s podrškom za dinamičke grafikone (Chart.js), praćenje tehničkih indikatora u realnom vremenu (preuzimanje s Yahoo Finance) te detaljnu tablicu evaluacijskih metrika.

---

## 📁 Struktura Projekta

```text
C:\ZavršniRad\
├── app.py                       # Flask poslužitelj i API krajnje točke
├── train_models.py              # Skripta za treniranje modela, inženjerstvo značajki i simulaciju
├── requirements.txt             # Popis biblioteka potrebnih za rad
├── skelet_zavrsnog_rada.md      # Upute i bilješke za rad
├── deployment_guide.md          # Upute za puštanje aplikacije u produkciju
├── zavrsni_rad.docx             # Izvorni akademski rad (Word format)
├── zavrsni_rad_corrected.docx   # Potpuno ispravljen i lektoriran rad spreman za predaju
├── templates/
│   └── index.html               # Predložak za premium web sučelje
└── outputs_v2/                  # Generirani modeli, skaleri i CSV tablice s rezultatima
    ├── thresholds.json          # Optimalni klasifikacijski pragovi po dionicama i modelima
    ├── top_features.csv         # Poredak značajki prema važnosti iz XGBoost-a
    ├── validation_metrics.csv   # Metrike performansi modela na validacijskom skupu
    ├── test_metrics_2024.csv    # Evaluacijske metrike na testnom skupu (2024.)
    ├── trading_simulation.csv   # Rezultati trading simulacije (Buy & Hold vs. Model)
    └── [TICKER]_[MODEL]...      # Serializirane datoteke modela (.joblib i .keras)
```

---

## 🛠️ Instalacija i Pokretanje

### 1. Preduvjeti

Osigurajte da imate instaliran Python (preporučeno v3.10 ili noviji).

### 2. Instalacija ovisnosti

Klonirajte ili kopirajte projekt u željeni direktorij te instalirajte sve potrebne pakete:

```bash
pip install -r requirements.txt
```

### 3. Treniranje modela (nije nužno ako su datoteke u `outputs_v2/` već prisutne)

Ako želite ponovno pokrenuti cijeli ML cjevovod, učitati svježe podatke, ponovno istrenirati modele i generirati nove tablice rezultata, pokrenite:

```bash
python train_models.py
```

*Napomena: Treniranje svih modela za 54 dionice s 10-godišnjim povijesnim podacima može potrajati nekoliko minuta ovisno o brzini vašeg procesora.*

### 4. Pokretanje web aplikacije

Pokrenite lokalni Flask poslužitelj:

```bash
python app.py
```

Nakon pokretanja, otvorite vaš web preglednik i posjetite:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📊 Rezultati i Evaluacija

Modeli su evaluirani na testnom skupu koji obuhvaća cjelokupnu 2024. godinu. Glavni rezultati pokazuju sljedeće:
1. **Povećanje stabilnosti:** Optimizacija klasifikacijskog praga dovela je do znatnog rasta Macro F1-Score-a, smanjujući udio lažnih signala rasta na volatilnom tržištu.
2. **Generiranje Alfe:** U trading simulaciji za 2024. godinu, aktivna strategija vođena strojnim učenjem uspješno je nadmašila pasivni Buy & Hold za većinu dionica (primjerice, kod **MSFT** i **AAPL** ostvaren je pozitivan diferencijal uz značajno smanjen drawdown).
3. **Značaj makroekonomskih podataka:** Poredak važnosti značajki ukazuje da promjene u indeksu volatilnosti **VIX** i prinosima indeksa **S&P 500** imaju izrazit utjecaj na predviđanje kratkoročnih trendova pojedinačnih dionica.

---

## 🎓 Akademska Usklađenost

Cjelokupni strojno-učenjački cjevovod u potpunosti je usklađen s metodologijom opisanom u ispravljenom akademskom radu (`zavrsni_rad_corrected.docx`). Sve modifikacije arhitekture LSTM mreže (32 LSTM jedinice, Dropout 0.3/0.2, skriveni Dense sloj sa 16 jedinica uz L2 regularizaciju, te treniranje kroz 20 epoha) sinkronizirane su između koda i teksta rada kako bi se osigurala besprijekorna znanstvena i tehnička dosljednost.
