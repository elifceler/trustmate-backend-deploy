TRUST_REASONING_PROMPT = r"""
Aşağıda bir satıcıya ait yorum analizlerinden ve profil verilerinden elde edilen özet metrikler yer almaktadır.
Bu verileri kullanarak satıcının genel güvenilirliği hakkında gerçekçi, çok boyutlu ve kullanıcıyı bilgilendiren teknik bir açıklama üret.

Kurallar:
- 2–4 cümlelik açıklama yaz.
- Satıcının güçlü ve zayıf yönlerini dengeli şekilde belirt.
- Eksik metrikleri veya yorum hacmini mutlaka değerlendir.
- Spam ihtimali veya yapaylık varsa mutlaka vurgula.
- JSON, başlık, markdown YOK — düz, akıcı ve bilgi veren metin üret.

Veri:
Satıcı adı: {seller_name}
Yorum sayısı: {review_count}
Metrikler:
- Teslimat: {delivery}
- Kalite: {quality}
- İade: {return_}
- Müşteri Hizmetleri: {customer_service}
- Duygu Skoru: {sentiment}
- Gerçeklik Skoru: {realness}
- Bonus Skor: {bonus}

Profil verileri:
- Ortalama Puan: {avg_rating}
- Yorumların Aynı Güne Yığılma Oranı: {same_day_ratio}
- Ortalama Fiyat Sapması: {price_deviation}
- Rakip Fiyat Ortalaması: {peer_avg_price}

Hazırladığın açıklama doğrudan kullanıcıya gösterilecektir. Gerçek bir karar destek sistemindeymiş gibi düşün ve mantıklı konuş.
"""
