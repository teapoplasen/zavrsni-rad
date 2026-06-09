# Vodič za Postavljanje Web Aplikacije na Poslužitelj (Render)

Pripremili smo vaš projekt **C:\ZavršniRad** za produkciju i uspješno ga inicijalizirali kao Git repozitorij. Kako bismo aplikaciju postavili na **Render** i dobili javni link, potrebno je pratiti korake u nastavku.

---

## 🛠️ Što smo napravili u projektu?

1. **Inicijaliziran je Git repozitorij** lokalno u mapi `C:\ZavršniRad` te su svi potrebni projektni dokumenti dodani u prvi commit na grani `main`.
2. **Dodana je `.gitignore` datoteka**: Sprječava dodavanje nepotrebnih datoteka na GitHub (kao što su privremeni Python i Jupyter predmemorijski zapisi, `.ipynb_checkpoints`, virtualna okruženja i sl.).
3. **Kreirana je `requirements.txt` datoteka**: Sadrži popis svih paketa potrebnih za pokretanje aplikacije. Posebno smo definirali:
   - **`tensorflow-cpu`** umjesto standardnog `tensorflow` kako bismo znatno smanjili veličinu aplikacije (slug size) i spriječili prekoračenje radne memorije (RAM) na Renderu.
   - **`gunicorn`** kao stabilan produkcijski web poslužitelj za Linux sustav koji koristi Render.
4. **Kreirana je `Procfile` datoteka**: Govori Renderu točnu naredbu za pokretanje aplikacije: `web: gunicorn app:app`.
5. **Optimizirano je učitavanje modela u app.py**:
   - Implementirali smo **predmemoriranje (caching)** strojno-učenih modela (`MODEL_CACHE`). Umjesto da se pri svakom API zahtjevu teški TensorFlow i Joblib modeli ponovno učitavaju s diska (što troši puno procesorskog vremena i uzrokuje curenje memorije na Renderu), modeli se sada učitavaju samo jednom pri prvom zahtjevu i čuvaju u memoriji (do najviše 40 modela istovremeno).

---

## 1. Korak: Postavljanje projekta na GitHub

Render se izravno povezuje s vašim GitHub računom i automatski povlači kod kada napravite izmjene.

1. Prijavite se na svoj **[GitHub](https://github.com)** račun u pregledniku.
2. U gornjem desnom kutu kliknite na **`+`** (plus) ikonu i odaberite **New repository** (Novi repozitorij).
3. Postavite sljedeće postavke:
   - **Repository name**: Upišite ime, npr. `zavrsni-rad` ili `stock-prediction-app`.
   - **Public/Private**: Odaberite po želji (preporučujemo *Private* ako želite da kod vidite samo vi i Render, ili *Public* ako ga želite pokazati mentorima/drugima).
   - **Nemojte** označiti opcije *Add a README file*, *Add .gitignore* ili *Choose a license* (jer smo te datoteke već pripremili lokalno).
4. Kliknite na gumb **Create repository**.
5. GitHub će vam prikazati stranicu s uputama pod naslovom **"…or push an existing repository from the command line"**. Kopirajte te naredbe. One izgledaju ovako:

```bash
git remote add origin https://github.com/VASE_KORISNICKO_IME/IME_REPOZITORIJA.git
git branch -M main
git push -u origin main
```

6. Otvorite svoj terminal (PowerShell ili Command Prompt) pozicioniran u mapi projekta (`C:\ZavršniRad`) i **pokrenite te naredbe**. 
   *(Ako se pojavi prozor za prijavu na GitHub, prijavite se putem preglednika ili unesite svoj GitHub Access Token).*

---

## 2. Korak: Deployment na Render

Nakon što je kod uspješno postavljen na GitHub, vrijeme je za Render:

1. Idite na **[Render Dashboard](https://dashboard.render.com)** i prijavite se.
2. Kliknite na plavi gumb **New +** u gornjem desnom kutu i odaberite **Web Service**.
3. Odaberite opciju **Build and deploy from a Git repository** (Povežite Render sa svojim GitHub računom ako već niste).
4. Pronađite svoj repozitorij (npr. `stock-prediction-app`) na popisu i kliknite **Connect**.
5. U postavkama web servisa postavite sljedeće parametre:
   - **Name**: Unesite naziv vaše aplikacije (to će biti dio vašeg javnog linka, npr. `moj-zavrsni-rad.onrender.com`).
   - **Region**: Odaberite regiju koja je najbliža (npr. *Frankfurt (Europe)*).
   - **Branch**: `main`
   - **Language**: `Python`
   - **Build Command**: `pip install -r requirements.txt` (Render će ovo automatski prepoznati).
   - **Start Command**: `gunicorn app:app` (Render će ovo automatski povući iz našeg `Procfile`-a).
6. Pomaknite se prema dolje i odaberite **Free** plan.
7. Kliknite na **Deploy Web Service**.

> [!NOTE]
> Prvi deployment može potrajati 5 do 10 minuta jer Render mora preuzeti TensorFlow, Pandas i ostale biblioteke te izgraditi okolinu. 
> 
> Jednom kada vidite zelenu poruku **"Your service is live!"**, vaša aplikacija je dostupna na linku prikazanom na vrhu Render stranice (ispod naziva projekta).

---

## 📈 Provjera rada na Renderu
Nakon što aplikacija bude aktivna, otvorite dodijeljeni link u pregledniku. 
- Aplikacija će raditi točno onako kako je radila lokalno na računalu.
- Svaki sljedeći put kada napravite promjenu u kodu i pokrenete `git push`, Render će automatski prepoznati promjenu i osvježiti vašu web aplikaciju na poslužitelju.
