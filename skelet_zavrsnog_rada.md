# Detaljan Kostur i Vodič za Pisanje Završnog Rada (cca 35-40 stranica)

Ovaj vodič predstavlja detaljan plan strukture vašeg završnog rada, podijeljen po poglavljima i procijenjenim stranicama, kako biste lakše rasporedili sadržaj i ispunili akademske zahtjeve Fakulteta elektrotehnike i računarstva (FER). 

Upute i natuknice su izravno povezane s vašim programskim kodom ([train_models.py](file:///C:/Zavr%C5%A1niRad/train_models.py)) i [app.py](file:///C:/Zavr%C5%A1niRad/app.py) te službenim predloškom ([zavrsni_rad (1).pdf](file:///C:/Zavr%C5%A1niRad/zavrsni_rad%20(1).pdf)).

---

## 📅 Raspodjela Stranica po Poglavljima (Ukupno: cca 35-40 str.)

1. **Uvodne stranice (str. 1–6)**: Naslovnica, Zadatak rada, Zahvala, Sažetak (HR/EN), Sadržaj.
2. **1. Uvod (str. 7–9)**: Motivacija, ciljevi, pregled literature.
3. **2. Teorijske osnove i odabrane tehnologije (str. 10–18)**: Teorija tržišta, algoritmi (Logistička regresija, Random Forest, XGBoost, LSTM), softverski stog (Python, Flask, TensorFlow, yfinance).
4. **3. Prikupljanje i priprema podataka (str. 19–26)**: Dohvaćanje podataka, inženjerstvo značajki (formule i objašnjenja indikatora), labeliranje, podjela na skupove, standardizacija.
5. **4. Izgradnja modela i optimizacija (str. 27–31)**: Treniranje modela, optimizacija klasifikacijskog praga (threshold tuning) pomoću Macro F1 metrike.
6. **5. Analiza performansi i rezultati (str. 32–37)**: Evaluacijske metrike, rezultati po sektorima, usporedna analiza.
7. **6. Razvoj web aplikacije (str. 38–41)**: Arhitektura sustava, Flask API, predmemoriranje (caching) za Render, vizualizacija.
8. **7. Zaključak i Literatura (str. 42–44)**: Rezime, budući rad, izvori.
9. **Prilozi (str. 45–46)**: Izjava o AI, Kod.

---

## ✍️ Detaljne Upute i Natuknice po Stranicama

### 1. UVODNE STRANICE (Stranice 1–6)
*   **Stranica 1: Naslovnica**
    *   *Sadržaj*: Slijediti točan predložak iz [zavrsni_rad (1).pdf](file:///C:/Zavr%C5%A1niRad/zavrsni_rad%20(1).pdf#page=1): Sveučilište u Zagrebu, FER, Završni rad br. XXXX, naslov rada (npr. *Sustav za kratkoročnu klasifikaciju kretanja cijena dionica primjenom metoda strojnog učenja*), vaše ime, Zagreb, lipanj, 2026.
*   **Stranica 2: Tekst zadatka**
    *   *Sadržaj*: Kopirati službeni tekst zadatka koji ste dobili od mentora (definiranje zadatka, obveza izrade web aplikacije, analiza performansi).
*   **Stranica 3: Zahvala**
    *   *Sadržaj*: Osobni osvrt. Zahvala mentoru, obitelji i kolegama.
*   **Stranica 4: Sažetak i ključne riječi (Hrvatski)**
    *   *Upute*: Napisati sažetak od 150 do 250 riječi. Objasniti problem (predviđanje kretanja dionica), metodologiju (XGBoost, LSTM, RF, LR s Yahoo Finance podacima) i rezultate (web aplikacija na Renderu).
    *   *Ključne riječi*: strojno učenje; klasifikacija cijena dionica; LSTM; XGBoost; Flask; Yahoo Finance.
*   **Stranica 5: Abstract and Keywords (Engleski)**
    *   *Upute*: Prijevod sažetka i ključnih riječi na engleski jezik (*Summary*, *Keywords*).
*   **Stranica 6: Sadržaj**
    *   *Upute*: Automatski generirani sadržaj (Table of Contents) s točnim brojevima stranica.

---

### 2. UVOD (Stranice 7–9)
*   **Stranica 7: Motivacija i problem predviđanja tržišta**
    *   *Natuknice*:
        *   Zašto je predviđanje kretanja dionica privlačno, ali iznimno teško (visoka volatilnost, šum u podacima, utjecaj vijesti).
        *   Razlika između fundamentalne analize (financijska izvješća) i tehničke analize (analiza grafikona i povijesnih cijena na čemu se temelji ovaj rad).
*   **Stranica 8: Pregled postojećih istraživanja i literatura**
    *   *Natuknice*:
        *   Tradicionalni statistički modeli (ARIMA, GARCH) naspram modernih metoda strojnog učenja.
        *   Uloga dubokog učenja (LSTM neuronskih mreža) u obradi vremenskih serija.
        *   Spomenuti nekoliko važnih akademskih radova koji se bave predviđanjem smjera kretanja dionica.
*   **Stranica 9: Cilj rada i struktura dokumenta**
    *   *Natuknice*:
        *   Što je točno cilj ovog rada: preuzimanje 54 dionice, generiranje tehničkih indikatora, izgradnja 4 klasifikacijska modela, razvoj interaktivne web aplikacije i objava na poslužitelju Render.
        *   Kratki opis što se nalazi u kojem poglavlju rada (od teorije do implementacije).

---

### 3. TEORIJSKE OSNOVE I ODABRANE TEHNOLOGIJE (Stranice 10–18)
*   **Stranica 10: Teorijska podloga tržišta kapitala**
    *   *Natuknice*:
        *   Hipoteza efikasnog tržišta (Efficient Market Hypothesis - EMH) Eugena Fame (slaba, polujaka i jaka forma).
        *   Kako se ovaj rad suprotstavlja EMH-u pretpostavkom da povijesni podaci i tehnički indikatori sadrže obrasce koji se mogu naučiti (slaba forma neefikasnosti).
*   **Stranica 11: Logistička regresija**
    *   *Natuknice*:
        *   Matematički model logističke regresije (sigmoidna funkcija, funkcija gubitka unakrsne entropije).
        *   Zašto je logistička regresija dobar "baseline" model (jednostavnost, interpretabilnost koeficijenata).
        *   Važnost standardizacije podataka za ovaj model.
*   **Stranica 12: Slučajne šume (Random Forest)**
    *   *Natuknice*:
        *   Stabla odluke (kriteriji podjele: Gini indeks, entropija).
        *   Koncept ansambla (Bagging - Bootstrap Aggregating).
        *   Kako Slučajne šume smanjuju varijancu pojedinačnih stabala i sprječavaju prenaučenost (*overfitting*).
*   **Stranica 13: XGBoost (Extreme Gradient Boosting)**
    *   *Natuknice*:
        *   Koncept Boosting-a (postupno dodavanje slabih učenika koji ispravljaju greške prethodnih).
        *   Zašto je XGBoost iznimno popularan i brz (regularizacija stakla, aproksimacija podjela, podrška za rijetke podatke).
        *   Značaj hiperparametara (stopa učenja, dubina stabla, regularizacijski faktori L1/L2).
*   **Stranica 14: Duboko učenje i Rekurentne neuronske mreže (RNN)**
    *   *Natuknice*:
        *   Arhitektura neuronskih mreža za vremenske serije.
        *   Problem nestajućeg i eksplodirajućeg gradijenta (vanishing/exploding gradient) kod standardnih RNN mreža prilikom obrade dužih sekvenci.
*   **Stranica 15: LSTM (Long Short-Term Memory) mreže**
    *   *Natuknice*:
        *   Arhitektura LSTM ćelije: stanje ćelije (*cell state*) i skriveno stanje (*hidden state*).
        *   Uloga vrata: zaboravna vrata (*forget gate*), ulazna vrata (*input gate*), izlazna vrata (*output gate*). Nacrtati ili opisati matematičke jednadžbe rada ovih vrata.
*   **Stranica 16: Softverski stog - Jezik i biblioteke za analizu podataka**
    *   *Natuknice*:
        *   Zašto Python (bogat ekosustav za znanost o podacima).
        *   **Pandas i NumPy**: manipulacija strukturiranim podacima i matrikama.
        *   **Scikit-learn**: implementacija logističke regresije, slučajnih šuma, skaliranja i evaluacijskih metrika.
*   **Stranica 17: Softverski stog - Duboko učenje i XGBoost**
    *   *Natuknice*:
        *   **TensorFlow i Keras**: izgradnja sekvencijalnih LSTM modela.
        *   **XGBoost biblioteka**: optimizirani gradijentni razvoj.
        *   **Joblib**: serijalizacija (spremanje) istreniranih modela i skalera na disk u svrhu kasnijeg učitavanja u web aplikaciji.
*   **Stranica 18: Softverski stog - Web poslužitelj i vizualizacija**
    *   *Natuknice*:
        *   **Flask**: lagani web okvir za Python, pogodan za izgradnju REST API-ja.
        *   **Chart.js**: JavaScript biblioteka za iscrtavanje responzivnih i interaktivnih grafikona na klijentskoj strani.
        *   **Gunicorn**: WSGI HTTP poslužitelj za Linux koji omogućuje stabilan rad Flask aplikacije u produkcijskom okruženju Renders poslužitelja.

---

### 4. PRIKUPUPLJANJE I PRIPREMA PODATAKA (Stranice 19–26)
*   **Stranica 19: Prikupljanje podataka i obuhvat dionica**
    *   *Natuknice*:
        *   Izvor podataka: **Yahoo Finance API** (korištenjem Python biblioteke `yfinance`).
        *   Obuhvat: 54 dionice raspoređene u 6 sektora (Tehnologija, Financije, Zdravstvo, Potrošačka dobra, Energija & Industrija, Komunikacije & usluge) - navesti nekoliko primjera (AAPL, MSFT, JPM, TSLA).
        *   Vremenski raspon trening skupa: **od 01.01.2015. do 01.01.2025.**
*   **Stranica 20: Inženjerstvo značajki (Feature Engineering) - Povrati i Pomični prosjeci**
    *   *Natuknice i Formule*:
        *   Jednodnevni i višednevni prinosi (1d, 5d, 10d, 20d):
            $$R_t^{(n)} = \frac{P_t - P_{t-n}}{P_{t-n}}$$
        *   Odnos trenutne cijene prema jednostavnim pomičnim prosjecima (SMA 5, 10, 20, 50):
            $$\text{SMA}_n = \frac{1}{n} \sum_{i=0}^{n-1} P_{t-i}, \quad \text{Omjer} = \frac{P_t}{\text{SMA}_n} - 1$$
*   **Stranica 21: Tehnički indikatori - MACD i RSI**
    *   *Natuknice i Formule*:
        *   **MACD (Moving Average Convergence Divergence)**: razlika između 12-dnevnog i 26-dnevnog eksponencijalnog pomičnog prosjeka (EMA). Signalna linija (9-dnevni EMA od MACD-a).
        *   **RSI (Relative Strength Index)**: mjera brzine i promjene kretanja cijene na bazi 14 dana:
            $$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}, \quad \text{RS} = \frac{\text{Prosječni dobitak}}{\text{Prosječni gubitak}}$$
*   **Stranica 22: Tehnički indikatori - Bollinger Bands i ATR**
    *   *Natuknice i Formule*:
        *   **Bollinger Bands**: Srednja linija (SMA 20) $\pm$ dvije standardne devijacije ($2\sigma$). U kodu se računa širina pojasa i trenutna pozicija unutar pojasa.
        *   **ATR (Average True Range)**: mjera volatilnosti tržišta na bazi 14 dana. True Range računa se kao maksimum triju razlika (High-Low, High-prev Close, Low-prev Close).
*   **Stranica 23: Volatilnost, Volumen i Momentum**
    *   *Natuknice*:
        *   Povijesna volatilnost na bazi 10 i 30 dana (standardna devijacija dnevnih povrata).
        *   Promjene volumena trgovanja (dnevni postotak promjene volumena i omjer trenutnog volumena prema 10-dnevnom prosjeku).
        *   Momentum normaliziran na cijenu prije 5 dana.
*   **Stranica 24: Makroekonomski pokazatelji (Market Context)**
    *   *Natuknice*:
        *   Zašto su u model uključeni globalni indeksi: **S&P 500** (`^GSPC`) kao reprezent općeg stanja na dioničkom tržištu i **VIX** (`^VIX`) kao indeks straha i volatilnosti.
        *   Ulazne značajke: dnevni povrat S&P 500 indeksa, zaključna cijena VIX-a i dnevni povrat VIX-a.
*   **Stranica 25: Labeliranje podataka i podjela na skupove**
    *   *Natuknice*:
        *   Kreiranje ciljne varijable (Label). Cilj je predvidjeti kretanje za sutrašnji dan ($T+1$). Ako je cijena zatvaranja sutra veća nego danas, oznaka je `1` (RAST), u suprotnom je `0` (PAD).
        *   Podjela podataka po vremenskom principu (Time-based split) kako bi se spriječilo curenje podataka iz budućnosti (*look-ahead bias*):
            *   **Trening skup**: 2015. - krajem 2022.
            *   **Validacijski skup**: 2023. godina (koristi se za Threshold Tuning i Early Stopping).
            *   **Testni skup**: 2024. godina (koristi se isključivo za konačnu evaluaciju performansi).
*   **Stranica 26: Standardizacija podataka (ManualStandardScaler)**
    *   *Natuknice i Formule*:
        *   Obrazložiti zašto je skaliranje nužno za Logističku regresiju i LSTM neuronsku mrežu (kako značajke s velikim rasponima ne bi nadvladale one s malim rasponima).
        *   Prikazati matematičku formulu standardizacije:
            $$Z = \frac{X - \mu}{\sigma}$$
        *   Objasniti ručnu implementaciju klase `ManualStandardScaler` koja računa aritmetičku sredinu ($\mu$) i standardnu devijaciju ($\sigma$) na trening skupu i primjenjuje ih na ostale skupove.

---

### 5. IZGRADNJA MODELA I OPTIMIZACIJA (Stranice 27–31)
*   **Stranica 27: Postavljanje eksperimenata i parametri klasičnih modela**
    *   *Natuknice*:
        *   Logistička regresija: korištenje `l2` regularizacije za sprječavanje prevelikih težina.
        *   Random Forest: konfiguracija stabala (broj procjenitelja = 100, maksimalna dubina).
        *   XGBoost Classifier: broj stabala, stopa učenja (`learning_rate`), maksimalna dubina pojedinog stabla.
*   **Stranica 28: Izgradnja LSTM modela**
    *   *Natuknice*:
        *   Konstrukcija 3D tenzora za LSTM: model prima povijesni prozor od 10 dana (`lookback = 10`) i 25 značajki (oblik ulaza: `[batch_size, 10, 25]`).
        *   Opisati funkciju [create_sequences](file:///C:/Zavr%C5%A1niRad/train_models.py#L101) koja ručno pretvara 2D tablične podatke u 3D sekvence.
*   **Stranica 29: Arhitektura neuronske mreže i proces treniranja**
    *   *Natuknice*:
        *   Slojevi LSTM modela (npr. LSTM sloj sa 64 jedinice, Dropout sloj od 0.2 za regularizaciju, Dense sloj sa sigmoid aktivacijom za klasifikaciju).
        *   Optimizator: Adam. Funkcija gubitka: binarna unakrsna entropija (*binary cross-entropy*).
        *   Korištenje `EarlyStopping` mehanizma na validacijskom skupu s parametrom `patience = 5` kako bi se spriječilo prenučavanje (overfitting).
*   **Stranica 30: Ručna optimizacija klasifikacijskog praga (Threshold Tuning)**
    *   *Natuknice*:
        *   Standardno pravilo klasifikacije svrstava vjerojatnosti $\ge 0.50$ u klasu 1 (rast). Međutim, financijske serije su asimetrične i neuravnotežene.
        *   Objasniti metodu [optimize_threshold](file:///C:/Zavr%C5%A1niRad/train_models.py#L111): pretražujemo pragove od $0.35$ do $0.65$ s korakom $0.01$ na validacijskom skupu i odabiremo onaj koji maksimizira **Macro F1-score**.
*   **Stranica 31: Zapis i pohrana optimalnih pragova**
    *   *Natuknice*:
        *   Kako se izračunati pragovi za svaku dionicu i model spremaju u `thresholds.json`.
        *   Objasniti važnost spremanja ovih pragova jer se oni kasnije učitavaju u Flask aplikaciji kako bi se donijela konačna odluka o prognozi smjera (`RASTE` / `PADA`) u realnom vremenu.

---

### 6. ANALIZA PERFORMANSI I REZULTATI (Stranice 32–37)
*   **Stranica 32: Evaluacijske metrike i ručne formule**
    *   *Natuknice i Formule*:
        *   Matrica zabune (Confusion Matrix): TP, TN, FP, FN.
        *   Točnost (Accuracy):
            $$\text{Točnost} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
        *   Preciznost (Precision) i Odziv (Recall) za klasu 1:
            $$\text{Preciznost} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Odziv} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
*   **Stranica 33: F1-score i ROC-AUC metrike**
    *   *Natuknice*:
        *   F1-score (harmonijska sredina preciznosti i odziva).
        *   Macro F1-score: prosjek F1-score-a za klasu 1 (rast) i klasu 0 (pad) - zašto je ovo najbolja metrika kod neuravnoteženih klasa.
        *   ROC krivulja i površina ispod krivulje (ROC-AUC) kao mjera sposobnosti modela da razlikuje klase bez obzira na prag.
*   **Stranica 34: Rezultati modela na testnom skupu (Opća tablica)**
    *   *Upute*:
        *   Prikazati tablicu prosječnih performansi svih modela na svim dionicama.
        *   Usporediti točnost i F1 mjeru za Logističku regresiju, Random Forest, XGBoost i LSTM.
*   **Stranica 35: Detaljna analiza performansi na primjeru odabranih dionica**
    *   *Natuknice*:
        *   Usporediti performanse na tehnološkom sektoru (npr. AAPL) u odnosu na stabilnije sektore (npr. Zdravstvo - JNJ ili Potrošačka dobra - PG).
        *   Koji modeli pokazuju najbolje rezultate na dionicama s visokim trendom, a koji na dionicama koje imaju visoku volatilnost (šum).
*   **Stranica 36: Usporedba klasičnih modela i LSTM neuronske mreže**
    *   *Natuknice*:
        *   Uočava li LSTM sekvencijalne obrasce bolje od klasičnih modela koji gledaju samo statički skup značajki?
        *   Prednosti XGBoost-a (brzina treniranja, stabilnost na tabličnim podacima) u odnosu na LSTM (zahtjevnost treniranja, osjetljivost na količinu podataka).
*   **Stranica 37: Rasprava o ograničenjima modela strojnog učenja u financijama**
    *   *Natuknice*:
        *   Problem stohastičke prirode burze (utjecaj geopolitike, vijesti i raspoloženja investitora koje tehnički indikatori ne mogu predvidjeti).
        *   Zašto točnost modela rijetko prelazi 55-58% (što je u financijskim predviđanjima zapravo izvrstan rezultat).

---

### 7. RAZVOJ WEB APLIKACIJE (Stranice 38–41)
*   **Stranica 38: Arhitektura i klijent-poslužitelj model**
    *   *Natuknice*:
        *   Dvoslojna arhitektura: **Flask poslužitelj** (Backend) i **HTML/CSS/JS sučelje** (Frontend).
        *   API komunikacija: klijent šalje asinkrone HTTP GET zahtjeve (korištenjem `fetch` API-ja u JS) prema poslužitelju, a poslužitelj vraća podatke u JSON formatu.
*   **Stranica 39: Implementacija API endpointa na poslužitelju**
    *   *Natuknice*:
        *   Opisati endpoint `/api/predict/<ticker>/<model_name>` koji preuzima live podatke, računa indikatore, učitava model i vraća predikciju.
        *   Opisati endpoint `/api/chart/<ticker>` koji vraća povijesne podatke za iscrtavanje grafikona.
*   **Stranica 40: Rješavanje problema prevelike potrošnje resursa (Model Caching)**
    *   *Natuknice*:
        *   Objasniti problem: učitavanje TensorFlow (.keras) i Joblib modela pri svakom API zahtjevu uzrokuje veliko kašnjenje (1-2 sekunde) i curenje memorije (OOM) na besplatnom poslužitelju Render (koji ima limit od 512MB RAM-a).
        *   Opisati implementaciju **caching mehanizma** ([MODEL_CACHE](file:///C:/Zavr%C5%A1niRad/app.py#L17-L29)) s ograničenjem na maksimalno 40 objekata u memoriji, što je riješilo problem sporosti i padanja aplikacije.
*   **Stranica 41: Korisničko sučelje i vizualizacija**
    *   *Natuknice*:
        *   Opis modernog i čistog tamnog sučelja (Bloomberg/Revolut stil).
        *   Kako se koristi **Chart.js** za prikaz interaktivnog grafikona kretanja dionice s gradijentnim ispunama.
        *   Prikaz performansi najboljeg modela u obliku grid kartica i tablica s evaluacijskim metrikama.

---

### 8. ZAKLJUČAK I LITERATURA (Stranice 42–44)
*   **Stranica 42: Rezime i zaključci rada**
    *   *Natuknice*:
        *   Sažeti ostvarene rezultate: uspješno trenirani modeli, izgrađena web aplikacija i puštena u rad na Render poslužitelju.
        *   Koji se model pokazao kao najstabilniji i najbrži za integraciju u sustave u realnom vremenu.
*   **Stranica 43: Smjernice za budući rad**
    *   *Natuknice*:
        *   Mogućnost proširenja skupa značajki analizom raspoloženja (Sentiment Analysis) s društvenih mreža (npr. X/Twitter) ili vijesti.
        *   Korištenje naprednijih arhitektura poput transformatora (Transformers - npr. Temporal Fusion Transformer).
        *   Uvođenje transakcijskih troškova i poreza u simulacije.
*   **Stranica 44: Literatura (References)**
    *   *Upute*: Navesti sve izvore prema FER-ovom IEEE standardu citiranja (knjige o strojnom učenju, znanstvene radove o financijskim vremenskim serijama, službene dokumentacije biblioteka TensorFlow, Scikit-learn, XGBoost, Flask i Yahoo Finance).

---

### 9. PRILOZI (Stranice 45–46)
*   **Stranica 45: Izjava o korištenju umjetne inteligencije**
    *   *Upute*: Popuniti točne podatke prema predlošku iz [zavrsni_rad (1).pdf](file:///C:/Zavr%C5%A1niRad/zavrsni_rad%20(1).pdf#page=13). Opisati da ste koristili Gemini/Antigravity AI asistenta za:
        1. Pisanje i optimizaciju HTML/CSS/JS koda u prednjem dijelu (frontend) aplikacije.
        2. Rješavanje problema s model cachingom i skaliranjem unutar Flask backend datoteke `app.py`.
        3. Pomoć u strukturiranju i planiranju kostura završnog rada.
*   **Stranica 46: Privitak A - Programski kod**
    *   *Upute*: Opisati strukturu priloženog koda (kako pokrenuti `train_models.py` za ponovno treniranje, kako pokrenuti `app.py` lokalno, i strukturu mapa u projektu).
