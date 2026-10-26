# TRELLIS-Silicon

Apple Silicon işlemcili Mac üzerinde tek bir görselden 3D model üretir. [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) modelini PyTorch MPS ile çalıştırır ve sonucu dokulu **GLB** dosyası olarak kaydeder. Terminalden veya yerel web arayüzünden kullanılabilir.

<p align="center">
  <img src="assets/demo.gif" width="720" alt="Girdi görseli ve üretilen 3D model">
</p>

Bu sürüm, [Sanchez Kim'in TRELLIS-Silicon projesi](https://github.com/sanchez-kim/trellis-silicon) üzerine kuruludur. Kaynak projelerin atıfları ve MIT lisansı korunur. Ayrıntılı teknik açıklamalar ve kaynak projenin performans ölçümleri için [İngilizce README](README.en.md) dosyasına bakabilirsiniz.

## 1. Önce hangi hesaplar ve girişler gerekiyor?

| Hizmet | Gerekli mi? | Ne için kullanılır? |
| --- | --- | --- |
| **Hugging Face** | **Evet** | Erişim izni isteyen model dosyalarını indirmek için. |
| **GitHub** | Yalnızca projeyi kendi hesabınızda yayımlamak için | Herkese açık depoyu indirmek ve yerelde çalıştırmak için giriş gerekmez. |
| **Yerel web arayüzü (Gradio)** | Hayır | Mac üzerinde açılır; ayrı kayıt veya giriş istemez. |

### Hugging Face hesabı ve model izinleri

Üretime başlamadan önce şu adımları tamamlayın:

1. [Hugging Face hesabı oluşturun](https://huggingface.co/join) veya [giriş yapın](https://huggingface.co/login).
2. Aynı hesap açıkken aşağıdaki iki model sayfasını ziyaret edin. Gösterilen koşulları okuyup kabul edin, istenen formu doldurun ve erişim isteğini gönderin.
   - [DINOv3 — facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m): görsel özelliklerini çıkarmak için kullanılır.
   - [RMBG-2.0 — briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0): arka planı kaldırmak için kullanılır.
3. Her iki model için erişimin açıldığından emin olun. İstek beklemedeyse onaylanmadan üretime geçmeyin. Yalnızca hesap açmak yeterli değildir.
4. Aşağıdaki kurulumu bitirdikten sonra terminalde `hf auth login` ile **erişim izni verilen aynı hesaba** giriş yapın.

Komut tarayıcı üzerinden giriş sunarsa o akışı tamamlayabilirsiniz. Token istenirse [Access Tokens](https://huggingface.co/settings/tokens) sayfasından **Read** yetkili bir token oluşturun ve giriş istemine yapıştırın. Model indirmek için Write yetkisi gerekmez. Git kimlik bilgilerine ekleme sorulursa `n` seçebilirsiniz.

Token'ı README'ye, kaynak koda veya GitHub'a eklemeyin. Bu proje için `.env` oluşturmanız gerekmez; `hf auth login` giriş bilgisini yerel Hugging Face önbelleğinde saklar.

> Model izinleri, terminal girişi ve model lisansı ayrı konulardır. RMBG-2.0 model sayfası ticari olmayan kullanım için erişim sunar; ticari kullanım için BRIA'nın lisans koşullarını inceleyin. Kodun MIT lisansı model ağırlıklarının koşullarını değiştirmez.

Resmî açıklamalar: [Hugging Face giriş komutları](https://huggingface.co/docs/huggingface_hub/guides/cli), [token yetkileri](https://huggingface.co/docs/hub/security-tokens), [model erişim izinleri](https://huggingface.co/docs/hub/models-gated).

## 2. Sistem gereksinimleri

- **Apple Silicon Mac:** M1 veya daha yeni işlemci ve MPS destekli macOS. Bu kurulum Intel Mac, Windows veya Linux için hazırlanmadı.
- **Python 3.11:** kilitli ortam ve CI bu sürümü kullanır. Paket Python 3.11 ve üzerini bildirir; başlangıç için 3.11 tercih edin.
- **Bellek:** 24 GB veya üzeri birleşik bellek önerilir. 16 GB cihazlarda çalışma garantisi verilmez.
- **Disk:** model dosyaları için yaklaşık 15 GB; Python ortamı ve derlenen bağımlılıklar için ayrıca boş alan gerekir.
- **İnternet:** kurulum, ilk model indirmeleri ve gerektiğinde ek çözünürlüklerin ağırlıklarını indirmek için.
- **Git ve Xcode Command Line Tools:** bağımlılıkları indirmek ve yerel uzantıları derlemek için.

Metal hızlandırmasının kullanılabilirliği macOS ve Xcode sürümüne bağlıdır. Metal uzantıları yüklenemezse proje daha yavaş alternatif arka uçları kullanabilir.

## 3. Kurulum

### Proje klasörünü açın

GitHub'daki kendi deponuzun **Code → HTTPS** adresini kopyalayarak klonlayın. Aşağıdaki yer tutucuyu kendi adresinizle değiştirin:

```bash
git clone https://github.com/KULLANICI_ADINIZ/trellis-silicon.git
cd trellis-silicon
```

Depo adı farklıysa `cd` komutunda o klasörün adını kullanın. ZIP indirdiyseniz ZIP'i açıp `setup.sh` dosyasının bulunduğu klasörde terminal açmanız yeterlidir.

Mevcut çalışma klasörünüz `3d-model-generator` ise asıl proje bir alt klasördedir:

```bash
cd trellis-silicon
```

### Geliştirme araçlarını kontrol edin

```bash
xcode-select --install
python3.11 --version
```

Command Line Tools zaten kuruluysa ilk komutu tekrar çalıştırmanız gerekmez. `python3.11` bulunamazsa Python 3.11'i kurup terminali yeniden açın. `uv` kuruluysa kurulum betiği Python ortamını onunla oluşturur; `uv` hesabı veya girişi gerekmez.

Metal ile daha hızlı doku işleme için uygun Xcode kurulumunda şu bileşeni indirebilirsiniz:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Bu komut mevcut Xcode kurulumunuzda desteklenmiyorsa aşağıdaki Metal'siz kurulum seçeneğini kullanabilirsiniz.

### Ortamı kurun, ardından giriş yapın

```bash
bash setup.sh
source .venv/bin/activate
hf auth login
hf auth whoami
```

`hf` komutu proje bağımlılıklarıyla kurulur; bu yüzden terminal girişi **kurulumdan sonra** yapılır. `setup.sh` sonunda giriş yapılmadığına dair uyarı görmeniz ilk kurulumda normaldir. `hf auth whoami` beklediğiniz hesabı göstermelidir; model izinlerini ayrıca iki model sayfasından kontrol edin.

Betik `.venv/` ortamını oluşturur, `requirements.lock` bağımlılıklarını kurar, kaynak bağımlılıkları `deps/` ve `TRELLIS.2/` içine indirir, Apple Silicon uyumluluk yamalarını uygular ve terminal komutlarını kaydeder. Model ağırlıkları ilk üretimde indirilir.

**Metal uzantılarını derlemeden kurmak için** yukarıdaki `bash setup.sh` yerine:

```bash
SKIP_METAL=1 bash setup.sh
```

Diğer adımlar aynıdır. Doku işleme daha yavaş olabilir.

## 4. Terminalden kullanım

Aşağıdaki komutları proje kökünde, yani `setup.sh` ve `assets/` klasörünün bulunduğu yerde çalıştırın. Her yeni terminalde önce ortamı etkinleştirin:

```bash
source .venv/bin/activate
```

### İlk örnek: vi.png

```bash
trellis-silicon ./assets/vi.png
```

Bu komut depodaki `assets/vi.png` görselini okur ve **komutu çalıştırdığınız klasöre `output_3d.glb`** kaydeder. Çıktı adı görsel adından otomatik türetilmez. Aynı çıktı adıyla tekrar üretim yaparsanız önceki dosyanın üzerine yazılır.

İlk çalıştırmada model indirmeleri nedeniyle bekleme süresi uzundur. Sonraki çalıştırmalar indirilen ağırlıkları yerel önbellekten kullanır. Üretim süresi Mac'in donanımına, sıcaklığına ve seçilen ayarlara bağlıdır.

### Çıktıları ayrı adlarla kaydetme

```bash
mkdir -p outputs
trellis-silicon ./assets/vi.png --output outputs/vi
trellis-silicon ./assets/bolet.png --output outputs/bolet
trellis-silicon ./assets/brighella_input.png --output outputs/brighella
```

Sonuçlar sırasıyla `outputs/vi.glb`, `outputs/bolet.glb` ve `outputs/brighella.glb` olur. `--output` değerine uzantı eklemeyin; hedef klasörü önce oluşturun. KDTree doku arka ucu kullanılırsa ek olarak `*_basecolor.png` kaydedilir.

### Kendi görselinizi kullanma

```bash
trellis-silicon "/tam/yol/benim gorselim.png" --output outputs/benim_modelim
```

Boşluk içeren yolları çift tırnak içine alın. Nesnenin net ve tamamının görünür olduğu, sade arka planlı bir görselle başlayın.

### Daha hızlı deneme

```bash
trellis-silicon ./assets/vi.png --steps 8 --output outputs/vi_fast
```

Daha az örnekleme adımı süreyi azaltabilir; sonuç kalitesi değişebilir.

### Daha yüksek çözünürlük

```bash
trellis-silicon ./assets/vi.png --pipeline-type 1024_cascade --texture-size 2048 --output outputs/vi_high
```

Bu seçenek daha fazla süre ve bellek gerektirebilir. GLB için varsayılan sadeleştirme hedefi 200.000 yüzdür. Daha fazla geometri korumak için:

```bash
BAKE_MAX_FACES=800000 trellis-silicon ./assets/vi.png --pipeline-type 1024_cascade --texture-size 2048 --output outputs/vi_detail
```

Yüksek yüz sayıları doku işleme maliyetini artırır; önce varsayılan ayarlarla başlayın.

### OBJ veya dokusuz GLB

```bash
# GLB yanında tam çözünürlüklü, dokusuz OBJ de kaydet
trellis-silicon ./assets/vi.png --obj --output outputs/vi_mesh

# Doku işlemeyi atla ve yalnızca geometri içeren GLB kaydet
trellis-silicon ./assets/vi.png --no-texture --output outputs/vi_geometry
```

### Komut seçenekleri

| Seçenek | Varsayılan | Açıklama |
| --- | --- | --- |
| `--output` | `output_3d` | Çıktı yolu/adı; uzantısız. |
| `--seed` | `42` | Rastgelelik tohumu. |
| `--pipeline-type` | `512` | `512`, `1024` veya `1024_cascade`. |
| `--texture-size` | `1024` | Doku boyutu: `512`, `1024` veya `2048`. |
| `--steps` | Model ayarı (genellikle 12) | Üç örnekleme aşamasındaki adım sayısı. |
| `--no-texture` | Kapalı | Doku işlemeyi atlar. |
| `--obj` | Kapalı | Sadeleştirme öncesi geometriyi ayrıca OBJ olarak kaydeder. |
| `--resident` | Kapalı | Modelleri MPS belleğinde tutar; bellek baskısı nedeniyle yavaşlatabilir. |

Tüm seçenekler: `trellis-silicon --help`.

## 5. Web arayüzü ve sonuçları açma

```bash
source .venv/bin/activate
trellis-silicon-web
```

Tarayıcıda **http://127.0.0.1:7860** adresini açın. Görsel yükleyin, ayarları seçin ve **Generate** düğmesine basın. İlk üretimde model yüklenir; ilerleme terminalde görünür. Aynı anda bir üretim çalışır.

Sonucu sayfadaki 3D görüntüleyicide inceleyebilirsiniz. Dosyalar `gradio_outputs/` klasörüne benzersiz adlarla kaydedilir. Arayüz yalnızca yerel adreste çalışır ve herkese açık paylaşım bağlantısı oluşturmaz. Durdurmak için terminalde `Ctrl+C` kullanın.

GLB dosyalarını Blender'da **File → Import → glTF 2.0 (.glb/.gltf)** yoluyla da açabilirsiniz.

## 6. Sık karşılaşılan sorunlar

| Sorun | Yapılacak işlem |
| --- | --- |
| `trellis-silicon: command not found` veya `hf: command not found` | Proje klasöründe `source .venv/bin/activate` çalıştırın. Kurulum yarıda kaldıysa `bash setup.sh` komutunu yeniden çalıştırın. |
| `401`, `403` veya `GatedRepoError` | `hf auth whoami` ile hesabı kontrol edin. İki model sayfasındaki erişim izinlerini ve token'ın okuma yetkisini doğrulayın. Gerekirse `hf auth login --force` ile yeniden giriş yapın. |
| Görsel bulunamadı | `assets/` klasörünün bulunduğu yerde olduğunuzu ve dosya yolunu kontrol edin. |
| Çıktı klasörü bulunamadı | Örneğin `mkdir -p outputs` çalıştırın; uygulama hedef klasörü otomatik oluşturmaz. |
| Metal derlemesi başarısız | `SKIP_METAL=1 bash setup.sh` ile uzantı derlemesini atlayın. Zaten kurulu uzantılar bu komutla kaldırılmaz. |
| Bellek yetersiz veya üretim çok yavaş | Diğer ağır uygulamaları kapatın, `512` çözünürlüğü kullanın ve `--resident` seçeneğini kapalı tutun. |
| Decoder boş mesh üretti / GPU watchdog hatası | `SPARSE_CONV_BACKEND=none trellis-silicon ./assets/vi.png` ile daha yavaş alternatif yolu deneyin. |
| `7860` portu kullanımda | Önceki `trellis-silicon-web` sürecini kendi terminalinde `Ctrl+C` ile kapatın. |

Üretim yalnızca çıkarım içindir; eğitim desteği yoktur. CUDA sürümündeki bazı mesh onarma adımları devre dışıdır ve çıktıda küçük delikler oluşabilir.

## 7. GitHub'a yükleme

**Git deposunun kökü `trellis-silicon/` klasörüdür.** `.venv/`, `deps/`, `TRELLIS.2/`, model önbellekleri, token dosyaları ve üretilen modeller depoya eklenmez. Örnek girdi görselleri `assets/` altında kalır. `.gitignore` bu ayrımı sağlar.

Bu çalışma kopyası kaynak deponun Git geçmişiyle gelir. Kendi GitHub hesabınızda boş bir depo oluşturduktan sonra mevcut uzak adresi kontrol edin:

```bash
git remote -v
```

`origin` hâlâ kaynak projeyi gösteriyorsa kendi adresinizle değiştirin:

```bash
git remote set-url origin https://github.com/KULLANICI_ADINIZ/trellis-silicon.git
```

Ardından dosyaları gözden geçirip yükleyin:

```bash
git status --short
git add .
git diff --cached --stat
git commit -m "docs: prepare repository for GitHub"
git push -u origin HEAD
```

`KULLANICI_ADINIZ` ve depo adını kendi bilgilerinizle değiştirin. Push aşamasında GitHub kimlik doğrulaması gerekir. GitHub hesabı ile Hugging Face hesabı birbirinden bağımsızdır. `.gitignore`, daha önce Git'e eklenmiş dosyaları otomatik olarak takipten çıkarmaz; commit öncesi eklenen dosyaları inceleyin.

## Geliştirme ve dosya yapısı

```text
assets/                  Örnek görseller ve dokümantasyon görselleri
src/trellis_silicon/      CLI, web arayüzü, üretim kodu ve uyumluluk yamaları
tests/                   Mesh çıkarımı testleri
tools/                   Profil ve önizleme araçları
.github/                 CI ve issue şablonları
setup.sh                 Kurulum betiği
requirements.lock        Sabitlenmiş Python bağımlılıkları
```

Katkı kuralları ve testler: [CONTRIBUTING.md](CONTRIBUTING.md). `TRELLIS.2/` kurulumda oluşturulur; kaynak değişiklikleri `src/trellis_silicon/patches/` içindeki yamalarla yapılır.

## Lisans ve kaynaklar

Bu depodaki kod [MIT lisansı](LICENSE) ile sunulur. Model ağırlıkları ayrı koşullara tabidir: [TRELLIS.2](https://github.com/microsoft/TRELLIS.2), [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0). Kendi girdi görsellerinizi ve çıktılarınızı paylaşırken ilgili kullanım haklarını da gözetin.

Bu çalışma [Sanchez Kim / trellis-silicon](https://github.com/sanchez-kim/trellis-silicon), [Shivam P Kumar / trellis-mac](https://github.com/shivampkumar/trellis-mac), Microsoft Research'ün TRELLIS.2 modeli ve [Pedro Naugusto'nun Metal araçları](https://github.com/pedronaugusto) üzerine kuruludur. Kaynak README'deki Brighella örneği [The Metropolitan Museum of Art koleksiyonundan](https://www.metmuseum.org/art/collection/search/201769) alınmış kamu malı (CC0) görseldir.
