from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'rahasia123'  # diperlukan untuk session
model = joblib.load('models/model_rf.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        nama = request.form.get('nama')

        ce_total = ro_total = ac_total = ae_total = 0

        for i in range(1, 13):
            ce_total += int(request.form.get(f'ce{i}', 0))
            ro_total += int(request.form.get(f'ro{i}', 0))
            ac_total += int(request.form.get(f'ac{i}', 0))
            ae_total += int(request.form.get(f'ae{i}', 0))

        ac_ce = ac_total - ce_total
        ae_ro = ae_total - ro_total
        noise = np.random.normal(0, 1)

        X_input = np.array([[ac_ce, ae_ro, noise]])
        prediksi = model.predict(X_input)[0]

        # simpan ke session
        session['hasil'] = {
            'nama': nama,
            'ce': ce_total,
            'ro': ro_total,
            'ac': ac_total,
            'ae': ae_total,
            'ac_ce': ac_ce,
            'ae_ro': ae_ro,
            'prediksi': prediksi
        }

        return redirect(url_for('hasil'))

    return render_template('index.html')

@app.route('/hasil')
def hasil():
    hasil = session.get('hasil', None)
    if not hasil:
        return redirect(url_for('index'))
    return render_template('hasil.html', **hasil)

if __name__ == '__main__':
    app.run(debug=True)
