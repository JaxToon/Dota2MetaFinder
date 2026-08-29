# Blueprint Sistem RL & Analisis Meta Dota 2 (Versi Revisi)

Dokumen ini adalah revisi dari blueprint awal, dengan tambahan: klarifikasi batasan teknis, estimasi compute, curriculum learning, reward shaping yang lebih matang, embedding untuk ketahanan patch, evaluasi/observability, dan status legal penggunaan.

---

## 0. Klarifikasi Scope & Status Legal (WAJIB dibaca sebelum lanjut)

Sebelum membangun apa pun, tentukan dulu tujuan penggunaan sistem ini, karena ini menentukan seluruh desain teknis di bawah:

| Skenario | Legal/ToS | Kompleksitas Teknis |
| --- | --- | --- |
| **A. Riset/sandbox offline** (custom lobby lokal, bot vs bot, tidak menyentuh match rated/publik) | Aman | Tetap tinggi tapi terkendali |
| **B. Bermain otomatis di match publik/ranked** | **Melanggar Valve Anti-Cheat/ToS — berisiko permanent ban** | Tidak relevan untuk dibahas |

Dokumen ini **hanya membahas Skenario A** (riset/sandbox). Jika tujuan Anda adalah B, hentikan proyek ini — tidak ada cara aman untuk melakukannya.

**Catatan tambahan**: akses low-level yang dipakai OpenAI Five (kontrol penuh atas seluruh unit, observation granular per-frame, bypass fog of war untuk training) diberikan Valve melalui **API khusus non-publik**. Bot Scripting API publik (folder `bots/` + VScript Lua) **tidak** menjamin akses selengkap itu. Langkah pertama proyek ini **bukan** menulis kode, melainkan:

1. Membuat prototipe kecil (1 hero, observation dasar: HP/posisi/cooldown) memakai Bot API publik.
2. Memverifikasi apakah data yang dibutuhkan (posisi unit musuh yang tidak terlihat langsung, cooldown ability musuh, dsb.) benar-benar tersedia.
3. Baru setelah validasi ini berhasil, lanjut ke desain observation space penuh di Bagian 3.

Jika validasi gagal (banyak data tidak accessible), rencana B adalah membangun **environment simulasi kustom** (bukan client Dota 2 asli) yang meniru mekanik inti — pendekatan yang lebih lambat dibuat tapi jauh lebih terkendali.

---

## 1. Konsep Utama & Tujuan Sistem

Sistem ini bertujuan menciptakan agen AI yang memiliki kesadaran penuh (*full awareness*) terhadap kemampuan hero, mekanik game, dan kondisi *match* Dota 2 — dilatih tidak hanya untuk menang, tapi untuk **menemukan meta baru, power spike, urutan item/skill efisien, dan taktik bertarung optimal**, di lingkungan sandbox/riset.

**Prinsip desain: mulai kecil, buktikan pipeline bekerja, baru naikkan kompleksitas.** Jangan desain langsung untuk 5v5 full map.

---

## 2. Arsitektur Teknis (Tech Stack & Pipeline Data)

| Komponen | Teknologi | Peran & Fungsi |
| --- | --- | --- |
| **Game Engine** | Dota 2 VScript (Lua) | Ekstraksi status unit, koordinat, cooldown, modifier setiap frame (target 30 Hz, **perlu diverifikasi stabil atau tidak**). |
| **Bridge Protocol** | TCP/UDP via ZeroMQ / Protobuf | Transfer payload JSON/Protobuf dari Dota 2 ke backend, latensi rendah. |
| **Data Ingestion Engine** | C++ / Rust | Parsing JSON → Contiguous Float Tensor, normalisasi data. |
| **Neural Network Core** | Python / PyTorch (LibTorch) | PPO + LSTM untuk memori temporal dan pengambilan keputusan. |
| **Analytics Engine** | Python (Pandas/NumPy) | Ekstraksi Value Head $V(s)$ untuk power spike, laning heatmap, efisiensi rotasi. |
| **Observability Stack** *(baru)* | TensorBoard / Weights & Biases | Monitoring reward curve, entropy, KL divergence, winrate vs checkpoint lama. |
| **Orchestration** *(baru)* | Ray RLlib / custom job queue | Mengelola banyak instance game paralel (rollout worker) untuk generate pengalaman training. |

---

## 3. Desain Model RL & Mekanisme Penemuan Meta

### A. Observation & Action Space

- **Observation Space (~20.000 float saat full scope):** HP/Mana relatif, posisi X/Y/Z, facing, cooldown ability/item, buff/debuff, koordinat & HP unit musuh/kawan yang terlihat.
  - **Catatan revisi:** angka ini adalah skala referensi OpenAI Five untuk 5v5 penuh. Untuk MVP (Bagian 6), observation space harus jauh lebih kecil (~500-2000 float) agar training bisa diverifikasi dalam waktu wajar.

- **Hierarchical Action Space:**
  1. **Action Type:** Move, Attack-Move, Cast Ability, Use Item, Stop/Hold.
  2. **Target Entity ID:** unit musuh/kawan yang ditargetkan.
  3. **Target Coordinates:** offset X, Y untuk skillshot/pergerakan.

### B. Embedding untuk Ketahanan Terhadap Hero/Item Baru *(baru — mengisi gap penting)*

Normalisasi rasio (Bagian 5) menangani **perubahan nilai stat**, tapi tidak menangani **perubahan dimensi** (hero baru, item baru, ability rework). Solusinya:

- Setiap **unit/hero** direpresentasikan sebagai **embedding vector** dari atribut dasarnya (tipe damage, role, base stats), bukan one-hot index tetap per hero. Hero baru cukup diberi vektor atribut baru — arsitektur network tidak perlu diubah.
- Setiap **item/ability** juga di-encode sebagai embedding dari efeknya (damage/heal/buff/mobility, target type, cooldown, mana cost), bukan indeks fixed-length dalam action space.
- Pendekatan ini mengikuti prinsip yang dipakai AlphaStar untuk unit type generalization — memungkinkan sistem tetap berjalan tanpa retrain total saat patch besar.

### C. Strategi Eksplorasi Meta

1. **Entropy Regularization** — menjaga keberagaman keputusan agar tidak terjebak satu trik statis.
2. **League Training (Self-Play & Exploiter Agents)** — 80% lawan diri sendiri, 20% lawan versi historis, untuk menemukan counter-strategy.
3. **Curiosity via RND** — dorongan eksplorasi area/aksi yang jarang dicoba.
4. **Annealing Curiosity Weight** *(baru)* — bobot RND diturunkan bertahap seiring training makin matang, supaya tidak bentrok dengan stabilitas league saat policy mulai konvergen. Tanpa ini, agent bisa terus "teralihkan" eksplorasi meski sudah punya strategi kuat.

### D. Reward Shaping *(direvisi — versi awal terlalu sederhana dan rawan reward hacking)*

Reward awal:
$$\text{Reward}_t = \Delta \text{NetWorth} + \Delta \text{Kills} - \Delta \text{Deaths} + \Delta \text{TowerHP Diff}$$

**Masalah**: agent bisa belajar farming pasif demi net worth tanpa niat menang, atau menghindari fight demi menjaga stat individu.

**Revisi:**

$$\text{Reward}_t = w_1 \Delta \text{NetWorth} + w_2 \Delta \text{Kills} - w_3 \Delta \text{Deaths} + w_4 \Delta \text{TowerHP} + w_5 \Delta \text{ObjectiveControl} + w_6 \cdot \mathbb{1}[\text{Win}]$$

- **ObjectiveControl**: mencakup Roshan control, ward placement/denial, map control (vision score).
- **Win indicator**: reward besar di akhir episode untuk menang, agar objektif jangka panjang tidak kalah oleh reward instan farming.
- **Team Spirit blending** (konsep dari OpenAI Five): reward individu di-blend dengan reward tim menggunakan parameter $\tau$ yang dinaikkan bertahap dari 0 (fokus individu di awal training, agar sinyal belajar lebih jelas) ke 1 (fokus tim penuh) seiring training matang.

$$\text{Reward}_{final} = (1-\tau) \cdot \text{Reward}_{individu} + \tau \cdot \text{Reward}_{tim}$$

---

## 4. Solusi Menghadapi Update Patch Game

1. **Feature Representation Abstraction** — data sebagai rasio dinormalisasi `[0.0–1.0]` (misal `Current_HP/Max_HP`, `Distance/Cast_Range`), bukan angka absolut.
2. **Embedding berbasis atribut** (lihat Bagian 3B) — menangani hero/item baru tanpa mengubah arsitektur.
3. **Transfer Learning (Fine-Tuning)** — checkpoint pre-patch dilatih ulang 1–3% dari durasi awal.
4. **Invalidasi League Archive** — refresh pool agen lawan historis yang masih pakai logika patch lama.
5. **Automated Patch-Diff Detection** *(baru)* — script yang membandingkan data hero/item versi baru vs lama, memberi notifikasi otomatis jika ada perubahan struktural (bukan cuma nilai) yang butuh perhatian manual sebelum fine-tuning dijalankan.

---

## 5. Curriculum Learning *(baru — bagian penting yang hilang di draf awal)*

Training langsung ke 5v5 full map hampir pasti gagal konvergen dalam waktu wajar. Urutan yang disarankan:

| Fase | Scope | Tujuan |
| --- | --- | --- |
| **Fase 0** | 1 hero, dummy target, tanpa musuh bergerak | Validasi pipeline: observation, action execution, reward signal jalan end-to-end |
| **Fase 1** | 1v1 mid lane, 3-5 hero, tanpa item kompleks | Validasi kemampuan dasar: last hit, deny, positioning, skill usage |
| **Fase 2** | 1v1 full (item lengkap, hingga level 25) | Validasi decision-making jangka menengah: build order, power spike |
| **Fase 3** | 2v2 / 3v3 di area terbatas | Validasi koordinasi tim skala kecil |
| **Fase 4** | 5v5 full map, hero pool terbatas | Validasi strategi makro: rotasi, objective, vision |
| **Fase 5** | 5v5 full hero pool | Scope penuh (opsional, tergantung ketersediaan compute) |

Setiap fase harus punya **kriteria kelulusan eksplisit** (misal: winrate >70% vs bot Valve level tertentu) sebelum naik ke fase berikutnya.

---

## 6. Estimasi Kebutuhan Compute *(baru — hilang total di draf awal)*

Sebagai referensi kasar (bukan janji hasil):

- **OpenAI Five** (5v5 full, referensi skala atas): ~128.000 CPU core + 256 GPU, berbulan-bulan training.
- **MVP realistis (Fase 0-2 di atas)**: puluhan hingga ratusan rollout worker paralel (instance Dota 2 headless jika memungkinkan, atau environment simulasi), 1-8 GPU untuk training, estimasi waktu training days-to-weeks tergantung kompleksitas.

**Rekomendasi**: tetapkan budget compute dan waktu maksimal per fase curriculum di atas sebelum mulai, dan evaluasi ulang kelayakan proyek jika Fase 0-1 saja sudah melebihi budget yang tersedia.

---

## 7. Evaluasi & Observability *(baru — hilang total di draf awal)*

Tanpa ini, training run yang berjalan berhari-hari tidak bisa didiagnosis kalau gagal.

- **Eval harness**: winrate terhadap checkpoint lama (untuk deteksi regresi), winrate vs bot Valve (baseline objektif), opsional vs manusia di sandbox privat.
- **Logging**: reward curve, policy entropy, KL divergence antar update, value loss — via TensorBoard/W&B.
- **Checkpoint versioning & rollback**: simpan checkpoint tiap N iterasi, dengan mekanisme rollback otomatis jika terdeteksi collapse (reward turun drastis/entropy collapse ke 0 tiba-tiba — pola umum di RL training).
- **A/B testing antar versi model** sebelum yang baru menggantikan liga utama.

---

## 8. Roadmap Implementasi Langkah-demi-Langkah (Revisi)

### Langkah 0: Validasi Batasan Teknis *(baru, wajib pertama)*
Bangun prototipe minimal untuk memverifikasi data apa saja yang benar-benar bisa diakses via Bot API publik. Tentukan apakah lanjut dengan client asli atau pivot ke simulator kustom.

### Langkah 1: Setup Lingkungan Dota 2
- Direktori Script: `<SteamLibrary>/steamapps/common/dota 2 beta/game/dota/scripts/vscripts/bots/`
- Steam Launch Options: `-dev -vconsole +dota_force_gamemode 1`

### Langkah 2: Lua Observation Harvester (`bot_generic.lua`)
Ambil status agen, cooldown, unit terdekat, ekspor ke JSON via socket. Mulai dengan observation minimal (Fase 0 curriculum).

### Langkah 3: C++ High-Speed Socket Receiver (`state_parser.cpp`)
Terima payload di port `5555`, parsing paralel, konversi ke tensor siap GPU.

### Langkah 4: Action Execution Handler (Lua)
Modul Lua yang menerima token aksi dari Python/C++ (`Action_UseAbilityOnEntity`, `Action_MoveToLocation`) dan mengeksekusinya presisi di engine.

### Langkah 5: PyTorch Neural Network Core (PPO + LSTM)
- Layer LSTM (mulai kecil, 128-256 unit untuk MVP; naik ke 1024-4096 hanya jika scope naik ke Fase 4-5).
- Policy Head (aksi), Value Head (prediksi $V(s)$).
- Embedding layer untuk hero/item (Bagian 3B).

### Langkah 6: Reward Shaping & Curriculum Setup
Implementasikan reward function revisi (Bagian 3D) dan susun training loop mengikuti urutan fase curriculum (Bagian 5).

### Langkah 7: Observability & Eval Harness *(baru)*
Setup logging (TensorBoard/W&B), checkpoint versioning, dan eval harness sebelum training skala besar dimulai — bukan setelah.

### Langkah 8: Engine Analisis Power-Spike & Meta Extractor
Script Python untuk mencatat perubahan Value Head $V(s)$ saat item dibeli/pertempuran terjadi, dirangkum jadi rekomendasi build, power spike, dan heatmap.

### Langkah 9: Patch Resilience Loop
Jalankan fine-tuning otomatis (Bagian 4) setiap kali patch baru terdeteksi, dengan patch-diff detection sebagai pemicu.

---

## Ringkasan Perubahan dari Draf Awal

| Area | Draf Awal | Revisi |
| --- | --- | --- |
| Validasi API | Tidak ada | Langkah 0 wajib sebelum desain final |
| Status legal | Tidak disebutkan | Dijelaskan eksplisit, scope dibatasi ke sandbox |
| Compute | Tidak dibahas | Estimasi + rekomendasi budget per fase |
| Scope training | Langsung tersirat 5v5 | Curriculum learning 6 fase eksplisit |
| Reward | 4 komponen sederhana | 6 komponen + team spirit blending |
| Ketahanan patch | Hanya normalisasi rasio | + embedding atribut + patch-diff detection |
| Evaluasi | Tidak ada | Eval harness, logging, checkpoint rollback |
| RND vs League | Disebut terpisah | Ditambah skema annealing agar tidak bentrok |
