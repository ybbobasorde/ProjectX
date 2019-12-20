from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/ai_ch02_1')
def ai_ch02_1():
    from ai_ch02.ai_ch02_1 import ai02_1_url1, ai02_plot_url1
    return render_template(
        'ai_ch02/ai_ch02_1.html', ai02_1_url1=ai02_1_url1, ai02_plot_url1=ai02_plot_url1
    )

@app.route('/ai_ch02_2')
def ai_ch02_2():
    from ai_ch02.ai_ch02_2 import ai02_2_url1, ai02_2_url2, ai02_2_url3, ai02_2_url4, ai02_2_url5, ai02_2_url6, ai02_2_url7, ai02_2_url8
    return render_template(
        'ai_ch02/ai_ch02_2.html', ai02_2_url1=ai02_2_url1, ai02_2_url2=ai02_2_url2, ai02_2_url3=ai02_2_url3, ai02_2_url4=ai02_2_url4, ai02_2_url5=ai02_2_url5,
            ai02_2_url6=ai02_2_url6, ai02_2_url7=ai02_2_url7, ai02_2_url8=ai02_2_url8
    )

@app.route('/ai_ch02_3')
def ai_ch02_3():
    from ai_ch02.ai_ch02_3 import ai02_3_url1, ai02_3_url2, ai02_3_url3
    return render_template(
        'ai_ch02/ai_ch02_3.html', ai02_3_url1=ai02_3_url1, ai02_3_url2=ai02_3_url2, ai02_3_url3=ai02_3_url3
    )

@app.route('/ai_ch02_4')
def ai_ch02_4():
    from ai_ch02.ai_ch02_4 import ai02_4_url1, ai02_4_url2
    return render_template(
        'ai_ch02/ai_ch02_4.html', ai02_4_url1=ai02_4_url1, ai02_4_url2=ai02_4_url2
    )

@app.route('/ai_ch02_5')
def ai_ch02_5():
    from ai_ch02.ai_ch02_5 import ai02_5_url1, ai02_5_url2, ai02_5_url3, ai02_5_url4, ai02_5_url5
    return render_template(
        'ai_ch02/ai_ch02_5.html', ai02_5_url1=ai02_5_url1, ai02_5_url2=ai02_5_url2, ai02_5_url3=ai02_5_url3, ai02_5_url4=ai02_5_url4, ai02_5_url5=ai02_5_url5
    )

@app.route('/ai_ch02_6')
def ai_ch02_6():
    from ai_ch02.ai_ch02_6 import ai02_plot_url4
    return render_template(
        'ai_ch02/ai_ch02_6.html', ai02_plot_url4=ai02_plot_url4
    )
    
@app.route('/ai_ch02_7')
def ai_ch02_7():
    from ai_ch02.ai_ch02_7 import ai02_plot_url2, ai02_plot_url3
    return render_template(
        'ai_ch02/ai_ch02_7.html', ai02_plot_url2=ai02_plot_url2, ai02_plot_url3=ai02_plot_url3
    )

@app.route('/ai_ch02_8')
def ai_ch02_8():
    from ai_ch02.ai_ch02_8 import ai02_8_url1, ai02_8_url2, ai02_8_url3, ai02_8_url4, ai02_8_url5, ai02_8_url6, ai02_8_url7
    return render_template(
        'ai_ch02/ai_ch02_8.html', ai02_8_url1=ai02_8_url1, ai02_8_url2=ai02_8_url2, ai02_8_url3=ai02_8_url3, ai02_8_url4=ai02_8_url4,
            ai02_8_url5=ai02_8_url5, ai02_8_url6=ai02_8_url6, ai02_8_url7=ai02_8_url7
    )

@app.route('/ai_ch02_9')
def ai_ch02_9():
    from ai_ch02.ai_ch02_9 import ai02_plot_url1, ai02_9_url1, ai02_9_url2, ai02_9_url3, ai02_9_url4, ai02_9_url5, ai02_9_url6
    return render_template(
        'ai_ch02/ai_ch02_9.html', ai02_plot_url1=ai02_plot_url1, ai02_9_url1=ai02_9_url1, ai02_9_url2=ai02_9_url2, ai02_9_url3=ai02_9_url3, ai02_9_url4=ai02_9_url4,
            ai02_9_url5=ai02_9_url5, ai02_9_url6=ai02_9_url6
    )

#
#
#

@app.route('/ai_ch03_1')
def ai_ch03_1():
    from ai_ch03.ai_ch03_1 import ai03_1_url1, ai03_1_url2, ai03_1_url3, ai03_1_url4, ai03_1_url5, ai03_1_url6, ai03_1_url7, ai03_1_url8, \
    ai03_1_plot_url1, ai03_1_plot_url2
    return render_template(
        'ai_ch03/ai_ch03_1.html', ai03_1_url1=ai03_1_url1, ai03_1_url2=ai03_1_url2, ai03_1_url3=ai03_1_url3, ai03_1_url4=ai03_1_url4, ai03_1_url5=ai03_1_url5,
        ai03_1_url6=ai03_1_url6, ai03_1_url7=ai03_1_url7, ai03_1_url8=ai03_1_url8, ai03_1_plot_url1=ai03_1_plot_url1, ai03_1_plot_url2=ai03_1_plot_url2
    )

@app.route('/ai_ch03_2')
def ai_ch03_2():
    from ai_ch03.ai_ch03_2 import ai03_2_url1, ai03_2_url2, ai03_2_url3, ai03_2_url4, ai03_2_url5, ai03_2_url6, ai03_2_url7, ai03_2_url8, \
    ai03_2_plot_url1, ai03_2_plot_url2
    return render_template(
        'ai_ch03/ai_ch03_2.html', ai03_2_url1=ai03_2_url1, ai03_2_url2=ai03_2_url2, ai03_2_url3=ai03_2_url3, ai03_2_url4=ai03_2_url4, ai03_2_url5=ai03_2_url5,
        ai03_2_url6=ai03_2_url6, ai03_2_url7=ai03_2_url7, ai03_2_url8=ai03_2_url8, ai03_2_plot_url1=ai03_2_plot_url1, ai03_2_plot_url2=ai03_2_plot_url2
    )

@app.route('/ai_ch03_3')
def ai_ch03_3():
    from ai_ch03.ai_ch03_3 import ai03_3_url1, ai03_3_url2, ai03_3_url3, \
    ai03_3_plot_url1
    return render_template(
        'ai_ch03/ai_ch03_3.html', ai03_3_url1=ai03_3_url1, ai03_3_url2=ai03_3_url2, ai03_3_url3=ai03_3_url3,
        ai03_3_plot_url1=ai03_3_plot_url1
    )

@app.route('/ai_ch03_4')
def ai_ch03_4():
    from ai_ch03.ai_ch03_4 import ai03_4_url1, ai03_4_url2, ai03_4_url3, ai03_4_url4, ai03_4_url5
    return render_template(
        'ai_ch03/ai_ch03_4.html', ai03_4_url1=ai03_4_url1, ai03_4_url2=ai03_4_url2, ai03_4_url3=ai03_4_url3, ai03_4_url4=ai03_4_url4, ai03_4_url5=ai03_4_url5
    )

@app.route('/ai_ch03_5')
def ai_ch03_5():
    from ai_ch03.ai_ch03_5 import ai03_5_url1, ai03_5_url2, ai03_5_url3, ai03_5_url4, ai03_5_url5, ai03_5_url6, ai03_5_url7, ai03_5_url8, ai03_5_url9, \
        ai03_5_plot_url1, ai03_5_plot_url2, ai03_5_plot_url3
    return render_template(
        'ai_ch03/ai_ch03_5.html', ai03_5_url1=ai03_5_url1, ai03_5_url2=ai03_5_url2, ai03_5_url3=ai03_5_url3, ai03_5_url4=ai03_5_url4, ai03_5_url5=ai03_5_url5,
            ai03_5_url6=ai03_5_url6, ai03_5_url7=ai03_5_url7, ai03_5_url8=ai03_5_url8, ai03_5_url9=ai03_5_url9,
            ai03_5_plot_url1=ai03_5_plot_url1, ai03_5_plot_url2=ai03_5_plot_url2, ai03_5_plot_url3=ai03_5_plot_url3
    )

@app.route('/ai_ch03_6')
def ai_ch03_6():
    from ai_ch03.ai_ch03_6 import ai03_6_url1, ai03_6_url2
    return render_template(
        'ai_ch03/ai_ch03_6.html', ai03_6_url1=ai03_6_url1, ai03_6_url2=ai03_6_url2
    )

#
#
#

@app.route('/ai_ch04_1')
def ai_ch04_1():
    from ai_ch04.ai_ch04_1 import ai04_1_url1, ai04_1_url2, \
    ai04_1_plot_url1
    return render_template(
        'ai_ch04/ai_ch04_1.html', ai04_1_url1=ai04_1_url1, ai04_1_url2=ai04_1_url2,
        ai04_1_plot_url1=ai04_1_plot_url1
    )

@app.route('/ai_ch04_3')
def ai_ch04_3():
    from ai_ch04.ai_ch04_3 import ai04_3_plot_url1
    return render_template(
        'ai_ch04/ai_ch04_3.html', ai04_3_plot_url1=ai04_3_plot_url1
    )

@app.route('/ai_ch04_4')
def ai_ch04_4():
    from ai_ch04.ai_ch04_4 import ai04_4_url1, ai04_4_url2, ai04_4_url3, ai04_4_url4, \
        ai04_4_plot_url1
    return render_template(
        'ai_ch04/ai_ch04_4.html', ai04_4_url1=ai04_4_url1, ai04_4_url2=ai04_4_url2, ai04_4_url3=ai04_4_url3, ai04_4_url4=ai04_4_url4,
            ai04_4_plot_url1=ai04_4_plot_url1
    )

@app.route('/ai_ch04_5')
def ai_ch04_5():
    from ai_ch04.ai_ch04_5 import ai04_5_url1, ai04_5_url2, \
        ai04_5_plot_url1
    return render_template(
        'ai_ch04/ai_ch04_5.html', ai04_5_url1=ai04_5_url1, ai04_5_url2=ai04_5_url2,
            ai04_5_plot_url1=ai04_5_plot_url1
    )

#
#
#

@app.route('/ai_ch05_3')
def ai_ch05_3():
    from ai_ch05.ai_ch05_3 import ai05_3_url1, \
        ai05_3_plot_url1, ai05_3_plot_url2
    return render_template(
        'ai_ch05/ai_ch05_3.html', ai05_3_url1=ai05_3_url1,
            ai05_3_plot_url1=ai05_3_plot_url1, ai05_3_plot_url2=ai05_3_plot_url2
    )

@app.route('/ai_ch05_5')
def ai_ch05_5():
    from ai_ch05.ai_ch05_5 import ai05_5_url1, \
        ai05_5_plot_url1, ai05_5_plot_url2, ai05_5_plot_url3, ai05_5_plot_url4
    return render_template(
        'ai_ch05/ai_ch05_5.html', ai05_5_url1=ai05_5_url1,
            ai05_5_plot_url1=ai05_5_plot_url1, ai05_5_plot_url2=ai05_5_plot_url2, ai05_5_plot_url3=ai05_5_plot_url3, ai05_5_plot_url4=ai05_5_plot_url4
    )

@app.route('/ai_ch05_6')
def ai_ch05_6():
    from ai_ch05.ai_ch05_6 import ai05_6_url1, ai05_6_url2, ai05_6_url3
    return render_template(
        'ai_ch05/ai_ch05_6.html', ai05_6_url1=ai05_6_url1, ai05_6_url2=ai05_6_url2, ai05_6_url3=ai05_6_url3
    )

#
#
#
   
@app.route('/ai_ch07_2')
def ai_ch07_2():
    from ai_ch07.ai_ch07_2 import ai07_2_url1
    return render_template(
        'ai_ch07/ai_ch07_2.html', ai07_2_url1=ai07_2_url1
    )

@app.route('/ai_ch07_3')
def ai_ch07_3():
    from ai_ch07.ai_ch07_3 import ai07_3_url1, ai07_3_url2, ai07_3_url3, ai07_3_url4, ai07_3_url5, ai07_3_url6, ai07_3_url7
    return render_template(
        'ai_ch07/ai_ch07_3.html', ai07_3_url1=ai07_3_url1, ai07_3_url2=ai07_3_url2, ai07_3_url3=ai07_3_url3, ai07_3_url4=ai07_3_url4, ai07_3_url5=ai07_3_url5,
        ai07_3_url6=ai07_3_url6, ai07_3_url7=ai07_3_url7
    )

@app.route('/ai_ch07_6')
def ai_ch07_6():
    from ai_ch07.ai_ch07_6 import ai07_6_url1
    return render_template(
        'ai_ch07/ai_ch07_6.html', ai07_6_url1=ai07_6_url1
    )

#
#
#
   
@app.route('/ai_ch10_6')
def ai_ch10_6():
    from ai_ch10.ai_ch10_6 import ai10_6_url1, ai10_6_url2, ai10_6_url3
    return render_template(
        'ai_ch10/ai_ch10_6.html', ai10_6_url1=ai10_6_url1, ai10_6_url2=ai10_6_url2, ai10_6_url3=ai10_6_url3
    )

#
#
#
   
@app.route('/ai_ch11_3')
def ai_ch11_3():
    from ai_ch11.ai_ch11_3 import ai11_plot_url1, ai11_plot_url2, ai11_plot_url3
    return render_template(
        'ai_ch11/ai_ch11_3.html', ai11_plot_url1=ai11_plot_url1, ai11_plot_url2=ai11_plot_url2, ai11_plot_url3=ai11_plot_url3
    )

@app.route('/ai_ch11_4')
def ai_ch11_4():
    from ai_ch11.ai_ch11_4 import ai11_plot_url1, ai11_plot_url2
    return render_template(
        'ai_ch11/ai_ch11_4.html', ai11_plot_url1=ai11_plot_url1, ai11_plot_url2=ai11_plot_url2
    )

@app.route('/ai_ch11_5')
def ai_ch11_5():
    from ai_ch11.ai_ch11_5 import ai11_5_url1, ai11_5_url2, ai11_5_url3, ai11_5_url4, ai11_5_url5, ai11_plot_url1, ai11_plot_url2
    return render_template(
        'ai_ch11/ai_ch11_5.html', ai11_5_url1=ai11_5_url1, ai11_5_url2=ai11_5_url2, ai11_5_url3=ai11_5_url3, ai11_5_url4=ai11_5_url4, ai11_5_url5=ai11_5_url5,
            ai11_plot_url1=ai11_plot_url1, ai11_plot_url2=ai11_plot_url2
    )

#
#
#
   
@app.route('/ai_ch12_2')
def ai_ch12_2():
    from ai_ch12.ai_ch12_2 import ai12_plot_url1
    return render_template(
        'ai_ch12/ai_ch12_2.html', ai12_plot_url1=ai12_plot_url1
    )

@app.route('/ai_ch12_3')
def ai_ch12_3():
    from ai_ch12.ai_ch12_3 import ai12_3_url1, ai12_3_url2, ai12_3_url3, ai12_plot_url1
    return render_template(
        'ai_ch12/ai_ch12_3.html', ai12_3_url1=ai12_3_url1, ai12_3_url2=ai12_3_url2, ai12_3_url3=ai12_3_url3, ai12_plot_url1=ai12_plot_url1
    )

@app.route('/ai_ch12_5')
def ai_ch12_5():
    from ai_ch12.ai_ch12_5 import ai12_plot_url1
    return render_template(
        'ai_ch12/ai_ch12_5.html', ai12_plot_url1=ai12_plot_url1
    )

#
#
#

@app.route('/ai_m01_ch01_1')
def ai_m01_ch01_1():
    from ai_m01_ch01.ai_m01_ch01_1 import ai01_1_url1, ai01_1_url2
    return render_template(
        'ai_m01_ch01/ai_m01_ch01_1.html', ai01_1_url1=ai01_1_url1, ai01_1_url2=ai01_1_url2
    )

@app.route('/ai_m01_ch02_2')
def ai_m01_ch02_2():
    from ai_m01_ch02.ai_m01_ch02_2 import ai02_2_url1, ai02_2_url2, ai02_2_url3, ai02_2_url4, ai02_2_url5, ai02_2_url6, ai02_2_url7, ai02_2_url8, ai02_2_url9, ai02_2_url10, ai02_2_url11
    return render_template(
        'ai_m01_ch02/ai_m01_ch02_2.html', ai02_2_url1=ai02_2_url1, ai02_2_url2=ai02_2_url2, ai02_2_url3=ai02_2_url3, ai02_2_url4=ai02_2_url4, ai02_2_url5=ai02_2_url5, ai02_2_url6=ai02_2_url6,
            ai02_2_url7=ai02_2_url7, ai02_2_url8=ai02_2_url8,ai02_2_url9=ai02_2_url9, ai02_2_url10=ai02_2_url10, ai02_2_url11=ai02_2_url11
    )

@app.route('/ai_m01_ch06_1')
def ai_m01_ch06_1():
    from ai_m01_ch06.ai_m01_ch06_1 import ai06_1_url1, ai06_1_url2, ai06_1_url3, ai06_1_url4, ai06_plot_url1
    return render_template(
        'ai_m01_ch06/ai_m01_ch06_1.html', ai06_1_url1=ai06_1_url1, ai06_1_url2=ai06_1_url2, ai06_1_url3=ai06_1_url3, ai06_1_url4=ai06_1_url4, ai06_plot_url1=ai06_plot_url1
    )

@app.route('/ai_m01_ch07_2')
def ai_m01_ch07_2():
    from ai_m01_ch07.ai_m01_ch07_2 import ai07_2_url1, ai07_2_url2, ai07_2_url3, ai07_2_url4, ai07_2_url5, ai07_plot_url1
    return render_template(
        'ai_m01_ch07/ai_m01_ch07_2.html', ai07_2_url1=ai07_2_url1, ai07_2_url2=ai07_2_url2, ai07_2_url3=ai07_2_url3, ai07_2_url4=ai07_2_url4, ai07_2_url5=ai07_2_url5, 
            ai07_plot_url1=ai07_plot_url1
    )

#
#
#
   
@app.route('/ml_ch02')
def ml_ch02():
    from ml_ch02.ml_ch02 import ml02_plot_url1, ml02_plot_url2, ml02_plot_url3, ml02_plot_url4, ml02_plot_url5, ml02_plot_url6, ml02_plot_url7, ml02_plot_url8
    return render_template(
        'ml_ch02/ml_ch02.html', ml02_plot_url1=ml02_plot_url1, ml02_plot_url2=ml02_plot_url2, ml02_plot_url3=ml02_plot_url3, ml02_plot_url4=ml02_plot_url4, 
        ml02_plot_url5=ml02_plot_url5, ml02_plot_url6=ml02_plot_url6, ml02_plot_url7=ml02_plot_url7, ml02_plot_url8=ml02_plot_url8
    )

@app.route('/ml_ch02_1')
def ml_ch02_1():
    from ml_ch02.ml_ch02_1 import ml02_plot_url1, ml02_plot_url2, ml02_plot_url3
    return render_template(
        'ml_ch02/ml_ch02_1.html', ml02_plot_url1=ml02_plot_url1, ml02_plot_url2=ml02_plot_url2, ml02_plot_url3=ml02_plot_url3
    )

@app.route('/ml_ch02_2')
def ml_ch02_2():
    from ml_ch02.ml_ch02_2 import ml02_plot_url4, ml02_plot_url5, ml02_plot_url6, ml02_plot_url7, ml02_plot_url8
    return render_template(
        'ml_ch02/ml_ch02_2.html', ml02_plot_url4=ml02_plot_url4, ml02_plot_url5=ml02_plot_url5, ml02_plot_url6=ml02_plot_url6, ml02_plot_url7=ml02_plot_url7, ml02_plot_url8=ml02_plot_url8
    )

@app.route('/ml_ch02_21')
def ml_ch02_21():
    from ml_ch02.ml_ch02_21 import ml02_plot_url4
    return render_template(
        'ml_ch02/ml_ch02_21.html', ml02_plot_url4=ml02_plot_url4
    )

@app.route('/ml_ch02_22')
def ml_ch02_22():
    from ml_ch02.ml_ch02_22 import ml02_plot_url5
    return render_template(
        'ml_ch02/ml_ch02_22.html', ml02_plot_url5=ml02_plot_url5
    )

@app.route('/ml_ch02_23')
def ml_ch02_23():
    from ml_ch02.ml_ch02_23 import ml02_plot_url6
    return render_template(
        'ml_ch02/ml_ch02_23.html', ml02_plot_url6=ml02_plot_url6
    )

@app.route('/ml_ch02_24')
def ml_ch02_24():
    from ml_ch02.ml_ch02_24 import ml02_plot_url7, ml02_plot_url8
    return render_template(
        'ml_ch02/ml_ch02_24.html', ml02_plot_url7=ml02_plot_url7, ml02_plot_url8=ml02_plot_url8
    )

#
#
#

@app.route('/ml_ch03')
def ml_ch03():
    from ml_ch03.ml_ch03 import ml03_url1, ml03_url2, ml03_url3, ml03_url4, ml03_url5, ml03_url6, ml03_url7, \
        ml03_plot_url1, ml03_plot_url2, ml03_plot_url3, ml03_plot_url4, ml03_plot_url5, ml03_plot_url6, ml03_plot_url7, ml03_plot_url8, \
        ml03_plot_url9, ml03_plot_url10, ml03_plot_url11, ml03_plot_url12, ml03_plot_url13, ml03_plot_url14, ml03_plot_url15
    return render_template(
        'ml_ch03/ml_ch03.html', ml03_url1=ml03_url1, ml03_url2=ml03_url2, ml03_url3=ml03_url3, ml03_url4=ml03_url4, ml03_url5=ml03_url5, 
        ml03_url6=ml03_url6, ml03_url7=ml03_url7, ml03_plot_url1=ml03_plot_url1, ml03_plot_url2=ml03_plot_url2, ml03_plot_url3=ml03_plot_url3, ml03_plot_url4=ml03_plot_url4,
        ml03_plot_url5=ml03_plot_url5, ml03_plot_url6=ml03_plot_url6, ml03_plot_url7=ml03_plot_url7, ml03_plot_url8=ml03_plot_url8, 
        ml03_plot_url9=ml03_plot_url9, ml03_plot_url10=ml03_plot_url10, ml03_plot_url11=ml03_plot_url11, ml03_plot_url12=ml03_plot_url12,
        ml03_plot_url13=ml03_plot_url13, ml03_plot_url14=ml03_plot_url14, ml03_plot_url15=ml03_plot_url15
    )

@app.route('/ml_ch03_0')
def ml_ch03_0():
    from ml_ch03.ml_ch03_0 import ml03_plot_url1
    return render_template(
        'ml_ch03/ml_ch03_0.html', ml03_plot_url1=ml03_plot_url1
    )

@app.route('/ml_ch03_1')
def ml_ch03_1():
    from ml_ch03.ml_ch03_1 import ml03_plot_url2, ml03_plot_url3, ml03_plot_url4, ml03_plot_url5, ml03_plot_url6
    return render_template(
        'ml_ch03/ml_ch03_1.html', ml03_plot_url2=ml03_plot_url2, ml03_plot_url3=ml03_plot_url3, ml03_plot_url4=ml03_plot_url4,
        ml03_plot_url5=ml03_plot_url5, ml03_plot_url6=ml03_plot_url6
    )

@app.route('/ml_ch03_11')
def ml_ch03_11():
    from ml_ch03.ml_ch03_11 import ml03_plot_url2
    return render_template(
        'ml_ch03/ml_ch03_11.html', ml03_plot_url2=ml03_plot_url2
    )

@app.route('/ml_ch03_12')
def ml_ch03_12():
    from ml_ch03.ml_ch03_12 import ml03_plot_url3, ml03_plot_url4
    return render_template(
        'ml_ch03/ml_ch03_12.html', ml03_plot_url3=ml03_plot_url3, ml03_plot_url4=ml03_plot_url4
    )

@app.route('/ml_ch03_13')
def ml_ch03_13():
    from ml_ch03.ml_ch03_13 import ml03_plot_url5
    return render_template(
        'ml_ch03/ml_ch03_13.html', ml03_plot_url5=ml03_plot_url5
    )

@app.route('/ml_ch03_14')
def ml_ch03_14():
    from ml_ch03.ml_ch03_14 import ml03_plot_url6
    return render_template(
        'ml_ch03/ml_ch03_14.html', ml03_plot_url6=ml03_plot_url6
    )

@app.route('/ml_ch03_2')
def ml_ch03_2():
    from ml_ch03.ml_ch03_2 import ml03_plot_url7, ml03_plot_url8
    return render_template(
        'ml_ch03/ml_ch03_2.html', ml03_plot_url7=ml03_plot_url7, ml03_plot_url8=ml03_plot_url8
    )

@app.route('/ml_ch03_21')
def ml_ch03_21():
    from ml_ch03.ml_ch03_21 import ml03_plot_url7
    return render_template(
        'ml_ch03/ml_ch03_21.html', ml03_plot_url7=ml03_plot_url7
    )

@app.route('/ml_ch03_22')
def ml_ch03_22():
    from ml_ch03.ml_ch03_22 import ml03_plot_url8
    return render_template(
        'ml_ch03/ml_ch03_22.html', ml03_plot_url8=ml03_plot_url8
    )

@app.route('/ml_ch03_3')
def ml_ch03_3():
    from ml_ch03.ml_ch03_3 import ml03_plot_url9, ml03_plot_url10, ml03_plot_url11
    return render_template(
        'ml_ch03/ml_ch03_3.html', ml03_plot_url9=ml03_plot_url9, ml03_plot_url10=ml03_plot_url10, ml03_plot_url11=ml03_plot_url11
    )

@app.route('/ml_ch03_4')
def ml_ch03_4():
    from ml_ch03.ml_ch03_4 import ml03_plot_url12, ml03_plot_url13, ml03_plot_url14
    return render_template(
        'ml_ch03/ml_ch03_4.html', ml03_plot_url12=ml03_plot_url12, ml03_plot_url13=ml03_plot_url13, ml03_plot_url14=ml03_plot_url14
    )

@app.route('/ml_ch03_41')
def ml_ch03_41():
    from ml_ch03.ml_ch03_41 import ml03_plot_url12
    return render_template(
        'ml_ch03/ml_ch03_41.html', ml03_plot_url12=ml03_plot_url12
    )

@app.route('/ml_ch03_42')
def ml_ch03_42():
    from ml_ch03.ml_ch03_42 import ml03_plot_url13
    return render_template(
        'ml_ch03/ml_ch03_42.html', ml03_plot_url13=ml03_plot_url13
    )

@app.route('/ml_ch03_43')
def ml_ch03_43():
    from ml_ch03.ml_ch03_43 import ml03_plot_url14
    return render_template(
        'ml_ch03/ml_ch03_43.html', ml03_plot_url14=ml03_plot_url14
    )

@app.route('/ml_ch03_5')
def ml_ch03_5():
    from ml_ch03.ml_ch03_5 import ml03_plot_url15
    return render_template(
        'ml_ch03/ml_ch03_5.html', ml03_plot_url15=ml03_plot_url15
    )

#
#
#

@app.route('/ml_ch04')
def ml_ch04():
    from ml_ch04.ml_ch04 import ml04_url1, ml04_url2, ml04_url3, ml04_url4, ml04_url5, ml04_url6, ml04_url7, ml04_url8, ml04_url9, ml04_url10, ml04_url11, ml04_url12, ml04_url13, \
        ml04_plot_url1, ml04_plot_url2, ml04_plot_url3
    return render_template(
        'ml_ch04/ml_ch04.html', ml04_url1=ml04_url1, ml04_url2=ml04_url2, ml04_url3=ml04_url3, ml04_url4=ml04_url4, ml04_url5=ml04_url5,
        ml04_url6=ml04_url6, ml04_url7=ml04_url7, ml04_url8=ml04_url8, ml04_url9=ml04_url9, ml04_url10=ml04_url10, ml04_url11=ml04_url11, ml04_url12=ml04_url12, ml04_url13=ml04_url13,
        ml04_plot_url1=ml04_plot_url1, ml04_plot_url2=ml04_plot_url2, ml04_plot_url3=ml04_plot_url3
    )

@app.route('/ml_ch04_1')
def ml_ch04_1():
    from ml_ch04.ml_ch04_1 import ml04_plot_url1, ml04_plot_url2
    return render_template(
        'ml_ch04/ml_ch04_1.html', ml04_plot_url1=ml04_plot_url1, ml04_plot_url2=ml04_plot_url2
    )

@app.route('/ml_ch04_11')
def ml_ch04_11():
    from ml_ch04.ml_ch04_11 import ml04_plot_url1
    return render_template(
        'ml_ch04/ml_ch04_11.html', ml04_plot_url1=ml04_plot_url1
    )

@app.route('/ml_ch04_12')
def ml_ch04_12():
    from ml_ch04.ml_ch04_12 import ml04_plot_url2
    return render_template(
        'ml_ch04/ml_ch04_12.html', ml04_plot_url2=ml04_plot_url2
    )

@app.route('/ml_ch04_2')
def ml_ch04_2():
    from ml_ch04.ml_ch04_2 import ml04_plot_url3
    return render_template(
        'ml_ch04/ml_ch04_2.html', ml04_plot_url3=ml04_plot_url3
    )

#
#
#

@app.route('/ml_ch05')
def ml_ch05():
    from ml_ch05.ml_ch05 import ml05_plot_url1, ml05_plot_url2, ml05_plot_url3, ml05_plot_url4, ml05_plot_url5, ml05_plot_url6, ml05_plot_url7, ml05_plot_url8, \
        ml05_plot_url9, ml05_plot_url10, ml05_plot_url11, ml05_plot_url12, ml05_plot_url13, ml05_plot_url14, ml05_plot_url15, ml05_plot_url16, \
        ml05_plot_url17, ml05_plot_url18
    return render_template(
        'ml_ch05/ml_ch05.html', ml05_plot_url1=ml05_plot_url1, ml05_plot_url2=ml05_plot_url2, ml05_plot_url3=ml05_plot_url3, ml05_plot_url4=ml05_plot_url4,
        ml05_plot_url5=ml05_plot_url5, ml05_plot_url6=ml05_plot_url6, ml05_plot_url7=ml05_plot_url7, ml05_plot_url8=ml05_plot_url8, 
        ml05_plot_url9=ml05_plot_url9, ml05_plot_url10=ml05_plot_url10, ml05_plot_url11=ml05_plot_url11, ml05_plot_url12=ml05_plot_url12,
        ml05_plot_url13=ml05_plot_url13, ml05_plot_url14=ml05_plot_url14, ml05_plot_url15=ml05_plot_url15, ml05_plot_url16=ml05_plot_url16,
        ml05_plot_url17=ml05_plot_url17, ml05_plot_url18=ml05_plot_url18
    )

@app.route('/ml_ch05_1')
def ml_ch05_1():
    from ml_ch05.ml_ch05_1 import ml05_plot_url1, ml05_plot_url2, ml05_plot_url3, ml05_plot_url4, ml05_plot_url5, ml05_plot_url6
    return render_template(
        'ml_ch05/ml_ch05_1.html', ml05_plot_url1=ml05_plot_url1, ml05_plot_url2=ml05_plot_url2, ml05_plot_url3=ml05_plot_url3, ml05_plot_url4=ml05_plot_url4,
        ml05_plot_url5=ml05_plot_url5, ml05_plot_url6=ml05_plot_url6
    )

@app.route('/ml_ch05_11')
def ml_ch05_11():
    from ml_ch05.ml_ch05_11 import ml05_plot_url1
    return render_template(
        'ml_ch05/ml_ch05_11.html', ml05_plot_url1=ml05_plot_url1
    )

@app.route('/ml_ch05_12')
def ml_ch05_12():
    from ml_ch05.ml_ch05_12 import ml05_plot_url2
    return render_template(
        'ml_ch05/ml_ch05_12.html', ml05_plot_url2=ml05_plot_url2
    )

@app.route('/ml_ch05_13')
def ml_ch05_13():
    from ml_ch05.ml_ch05_13 import ml05_plot_url3, ml05_plot_url4, ml05_plot_url5, ml05_plot_url6
    return render_template(
        'ml_ch05/ml_ch05_13.html', ml05_plot_url3=ml05_plot_url3, ml05_plot_url4=ml05_plot_url4,
        ml05_plot_url5=ml05_plot_url5, ml05_plot_url6=ml05_plot_url6
    )

@app.route('/ml_ch05_2')
def ml_ch05_2():
    from ml_ch05.ml_ch05_2 import ml05_plot_url7, ml05_plot_url8, ml05_plot_url9, ml05_plot_url10
    return render_template(
        'ml_ch05/ml_ch05_2.html', ml05_plot_url7=ml05_plot_url7, ml05_plot_url8=ml05_plot_url8, ml05_plot_url9=ml05_plot_url9, ml05_plot_url10=ml05_plot_url10
    )

@app.route('/ml_ch05_21')
def ml_ch05_21():
    from ml_ch05.ml_ch05_21 import ml05_plot_url7
    return render_template(
        'ml_ch05/ml_ch05_21.html', ml05_plot_url7=ml05_plot_url7
    )

@app.route('/ml_ch05_22')
def ml_ch05_22():
    from ml_ch05.ml_ch05_22 import ml05_plot_url8
    return render_template(
        'ml_ch05/ml_ch05_22.html', ml05_plot_url8=ml05_plot_url8
    )

@app.route('/ml_ch05_23')
def ml_ch05_23():
    from ml_ch05.ml_ch05_23 import ml05_plot_url9, ml05_plot_url10
    return render_template(
        'ml_ch05/ml_ch05_23.html', ml05_plot_url9=ml05_plot_url9, ml05_plot_url10=ml05_plot_url10
    )

@app.route('/ml_ch05_3')
def ml_ch05_3():
    from ml_ch05.ml_ch05_3 import ml05_plot_url11, ml05_plot_url12, ml05_plot_url13, ml05_plot_url14, ml05_plot_url15, ml05_plot_url16, \
        ml05_plot_url17, ml05_plot_url18
    return render_template(
        'ml_ch05/ml_ch05_3.html', ml05_plot_url11=ml05_plot_url11, ml05_plot_url12=ml05_plot_url12,
        ml05_plot_url13=ml05_plot_url13, ml05_plot_url14=ml05_plot_url14, ml05_plot_url15=ml05_plot_url15, ml05_plot_url16=ml05_plot_url16,
        ml05_plot_url17=ml05_plot_url17, ml05_plot_url18=ml05_plot_url18
    )

@app.route('/ml_ch05_31')
def ml_ch05_31():
    from ml_ch05.ml_ch05_31 import ml05_plot_url11, ml05_plot_url12, ml05_plot_url13
    return render_template(
        'ml_ch05/ml_ch05_31.html', ml05_plot_url11=ml05_plot_url11, ml05_plot_url12=ml05_plot_url12,
        ml05_plot_url13=ml05_plot_url13
    )

@app.route('/ml_ch05_32')
def ml_ch05_32():
    from ml_ch05.ml_ch05_32 import ml05_plot_url14, ml05_plot_url15, ml05_plot_url16
    return render_template(
        'ml_ch05/ml_ch05_32.html', ml05_plot_url14=ml05_plot_url14, ml05_plot_url15=ml05_plot_url15, ml05_plot_url16=ml05_plot_url16
    )

@app.route('/ml_ch05_33')
def ml_ch05_33():
    from ml_ch05.ml_ch05_33 import ml05_plot_url17
    return render_template(
        'ml_ch05/ml_ch05_33.html', ml05_plot_url17=ml05_plot_url17
    )

@app.route('/ml_ch05_34')
def ml_ch05_34():
    from ml_ch05.ml_ch05_34 import ml05_plot_url18
    return render_template(
        'ml_ch05/ml_ch05_34.html', ml05_plot_url18=ml05_plot_url18
    )

#
#
#

@app.route('/ml_ch06')
def ml_ch06():
    from ml_ch06.ml_ch06 import ml06_url1, ml06_url2, ml06_url3, ml06_url4, ml06_url5, ml06_url6, ml06_url7, ml06_url8, ml06_url9, ml06_url10, \
        ml06_url11, ml06_url12, ml06_url13, ml06_url14, ml06_url15, ml06_url16, ml06_url17, ml06_plot_url1, ml06_plot_url2, ml06_plot_url3, ml06_plot_url4
    return render_template(
        'ml_ch06/ml_ch06.html', ml06_url1=ml06_url1, ml06_url2=ml06_url2, ml06_url3=ml06_url3, ml06_url4=ml06_url4, ml06_url5=ml06_url5, ml06_url6=ml06_url6, ml06_url7=ml06_url7, 
        ml06_url8=ml06_url8, ml06_url9=ml06_url9, ml06_url10=ml06_url10, ml06_url11=ml06_url11, ml06_url12=ml06_url12, ml06_url13=ml06_url13,
        ml06_url14=ml06_url14, ml06_url15=ml06_url15, ml06_url16=ml06_url16, ml06_url17=ml06_url17,
        ml06_plot_url1=ml06_plot_url1, ml06_plot_url2=ml06_plot_url2, ml06_plot_url3=ml06_plot_url3, ml06_plot_url4=ml06_plot_url4
    )

@app.route('/ml_ch06_1')
def ml_ch06_1():
    from ml_ch06.ml_ch06_1 import ml06_url1
    return render_template(
        'ml_ch06/ml_ch06_1.html', ml06_url1=ml06_url1
    )

@app.route('/ml_ch06_2')
def ml_ch06_2():
    from ml_ch06.ml_ch06_2 import ml06_url2, ml06_url3, ml06_url4, ml06_url5
    return render_template(
        'ml_ch06/ml_ch06_2.html', ml06_url2=ml06_url2, ml06_url3=ml06_url3, ml06_url4=ml06_url4, ml06_url5=ml06_url5
    )

@app.route('/ml_ch06_3')
def ml_ch06_3():
    from ml_ch06.ml_ch06_3 import ml06_plot_url1, ml06_plot_url2
    return render_template(
        'ml_ch06/ml_ch06_3.html', ml06_plot_url1=ml06_plot_url1, ml06_plot_url2=ml06_plot_url2
    )

@app.route('/ml_ch06_4')
def ml_ch06_4():
    from ml_ch06.ml_ch06_4 import ml06_url6, ml06_url7, ml06_url8, ml06_url9, ml06_url10
    return render_template(
        'ml_ch06/ml_ch06_4.html', ml06_url6=ml06_url6, ml06_url7=ml06_url7, ml06_url8=ml06_url8, ml06_url9=ml06_url9, ml06_url10=ml06_url10
    )

@app.route('/ml_ch06_5')
def ml_ch06_5():
    from ml_ch06.ml_ch06_5 import ml06_url11, ml06_url12, ml06_url13, ml06_url14, ml06_url15, ml06_plot_url3
    return render_template(
        'ml_ch06/ml_ch06_5.html', ml06_url11=ml06_url11, ml06_url12=ml06_url12, ml06_url13=ml06_url13, ml06_url14=ml06_url14, ml06_url15=ml06_url15,
        ml06_plot_url3=ml06_plot_url3
    )

@app.route('/ml_ch06_6')
def ml_ch06_6():
    from ml_ch06.ml_ch06_6 import ml06_url16, ml06_url17, ml06_plot_url4
    return render_template(
        'ml_ch06/ml_ch06_6.html', ml06_url16=ml06_url16, ml06_url17=ml06_url17, ml06_plot_url4=ml06_plot_url4
    )

#
#
#

@app.route('/ml_ch07')
def ml_ch07():
    from ml_ch07.ml_ch07 import ml07_url1, ml07_url2, ml07_url3, ml07_url4, ml07_url5, ml07_url6, ml07_url7, ml07_url8, ml07_url9, ml07_url10, \
        ml07_plot_url1, ml07_plot_url2, ml07_plot_url3, ml07_plot_url4, ml07_plot_url5
    return render_template(
        'ml_ch07/ml_ch07.html', ml07_url1=ml07_url1, ml07_url2=ml07_url2, ml07_url3=ml07_url3, ml07_url4=ml07_url4, ml07_url5=ml07_url5, ml07_url6=ml07_url6,
        ml07_url7=ml07_url7, ml07_url8=ml07_url8, ml07_url9=ml07_url9, ml07_url10=ml07_url10,  
        ml07_plot_url1=ml07_plot_url1, ml07_plot_url2=ml07_plot_url2, ml07_plot_url3=ml07_plot_url3, ml07_plot_url4=ml07_plot_url4, ml07_plot_url5=ml07_plot_url5
    )

@app.route('/ml_ch07_1')
def ml_ch07_1():
    from ml_ch07.ml_ch07_1 import ml07_plot_url1
    return render_template(
        'ml_ch07/ml_ch07_1.html', ml07_plot_url1=ml07_plot_url1
    )

@app.route('/ml_ch07_2')
def ml_ch07_2():
    from ml_ch07.ml_ch07_2 import ml07_url1, ml07_url2, ml07_url3
    return render_template(
        'ml_ch07/ml_ch07_2.html', ml07_url1=ml07_url1, ml07_url2=ml07_url2, ml07_url3=ml07_url3
    )

@app.route('/ml_ch07_3')
def ml_ch07_3():
    from ml_ch07.ml_ch07_3 import ml07_url4, ml07_url5, ml07_url6, ml07_plot_url2, ml07_plot_url3
    return render_template(
        'ml_ch07/ml_ch07_3.html', ml07_url4=ml07_url4, ml07_url5=ml07_url5, ml07_url6=ml07_url6, ml07_plot_url2=ml07_plot_url2, ml07_plot_url3=ml07_plot_url3
    )

@app.route('/ml_ch07_4')
def ml_ch07_4():
    from ml_ch07.ml_ch07_4 import ml07_url7, ml07_url8, ml07_plot_url4
    return render_template(
        'ml_ch07/ml_ch07_4.html', ml07_url7=ml07_url7, ml07_url8=ml07_url8, ml07_plot_url4=ml07_plot_url4
    )

@app.route('/ml_ch07_5')
def ml_ch07_5():
    from ml_ch07.ml_ch07_5 import ml07_url9, ml07_url10, ml07_plot_url5
    return render_template(
        'ml_ch07/ml_ch07_5.html', ml07_url9=ml07_url9, ml07_url10=ml07_url10, ml07_plot_url5=ml07_plot_url5
    )

#
#
#

#
#
#

#from ml_ch09.ml_ch09 import ReviewForm, classify, train, sqlite_entry, db
from ml_ch09.ml_ch09 import ReviewForm, classify, sqlite_entry, db

@app.route('/ml_ch09')
def ml_ch09():
    form = ReviewForm(request.form)
    return render_template('ml_ch09/reviewform.html', form=form)

@app.route('/ml_ch09results', methods=['POST'])
def ml_ch09results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('ml_ch09/results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('ml_ch09/reviewform.html', form=form)

@app.route('/ml_ch09thanks', methods=['POST'])
def ml_ch09feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    #train(review, y)
    sqlite_entry(db, review, y)
    return render_template('ml_ch09/thanks.html')


#
#
#

@app.route('/ml_ch10')
def ml_ch10():
    from ml_ch10.ml_ch10 import ml10_url1, ml10_url2, ml10_url3, ml10_url4, ml10_url5, ml10_url6, ml10_url7, ml10_url8, ml10_url9, ml10_url10, ml10_url11, ml10_url12, \
        ml10_url13, ml10_url14, ml10_url15, ml10_url16, ml10_url17, ml10_url18, \
        ml10_plot_url1, ml10_plot_url2, ml10_plot_url3, ml10_plot_url4, ml10_plot_url5, ml10_plot_url6, ml10_plot_url7, ml10_plot_url8, ml10_plot_url9, ml10_plot_url10, \
        ml10_plot_url11, ml10_plot_url12
    return render_template(
        'ml_ch10/ml_ch10.html', ml10_url1=ml10_url1, ml10_url2=ml10_url2, ml10_url3=ml10_url3, ml10_url4=ml10_url4, ml10_url5=ml10_url5, ml10_url6=ml10_url6, ml10_url7=ml10_url7, 
        ml10_url8=ml10_url8, ml10_url9=ml10_url9, ml10_url10=ml10_url10, ml10_url11=ml10_url11, ml10_url12=ml10_url12, ml10_url13=ml10_url13, ml10_url14=ml10_url14,
        ml10_url15=ml10_url15, ml10_url16=ml10_url16, ml10_url17=ml10_url17, ml10_url18=ml10_url18,
        ml10_plot_url1=ml10_plot_url1, ml10_plot_url2=ml10_plot_url2, ml10_plot_url3=ml10_plot_url3, ml10_plot_url4=ml10_plot_url4, ml10_plot_url5=ml10_plot_url5,
        ml10_plot_url6=ml10_plot_url6, ml10_plot_url7=ml10_plot_url7, ml10_plot_url8=ml10_plot_url8, ml10_plot_url9=ml10_plot_url9, ml10_plot_url10=ml10_plot_url10,
        ml10_plot_url11=ml10_plot_url11, ml10_plot_url12=ml10_plot_url12
    )
    
@app.route('/ml_ch10_1')
def ml_ch10_1():
    from ml_ch10.ml_ch10_1 import ml10_plot_url1, ml10_plot_url2
    return render_template(
        'ml_ch10/ml_ch10_1.html', ml10_plot_url1=ml10_plot_url1, ml10_plot_url2=ml10_plot_url2
    )

@app.route('/ml_ch10_2')
def ml_ch10_2():
    from ml_ch10.ml_ch10_2 import ml10_url1, ml10_url2, ml10_url3, ml10_url4, ml10_url5, ml10_url6, ml10_url7, \
        ml10_plot_url3, ml10_plot_url4, ml10_plot_url5
    return render_template(
        'ml_ch10/ml_ch10_2.html', ml10_url1=ml10_url1, ml10_url2=ml10_url2, ml10_url3=ml10_url3, ml10_url4=ml10_url4, ml10_url5=ml10_url5, ml10_url6=ml10_url6, ml10_url7=ml10_url7,
        ml10_plot_url3=ml10_plot_url3, ml10_plot_url4=ml10_plot_url4, ml10_plot_url5=ml10_plot_url5
    )

@app.route('/ml_ch10_3')
def ml_ch10_3():
    from ml_ch10.ml_ch10_3 import ml10_url8, ml10_url9, ml10_plot_url6
    return render_template(
        'ml_ch10/ml_ch10_3.html', ml10_url8=ml10_url8, ml10_url9=ml10_url9, ml10_plot_url6=ml10_plot_url6
    )

@app.route('/ml_ch10_4')
def ml_ch10_4():
    from ml_ch10.ml_ch10_4 import ml10_url10, ml10_url11, ml10_plot_url7
    return render_template(
        'ml_ch10/ml_ch10_4.html', ml10_url10=ml10_url10, ml10_url11=ml10_url11, ml10_plot_url7=ml10_plot_url7
    )

@app.route('/ml_ch10_5')
def ml_ch10_5():
    from ml_ch10.ml_ch10_5 import ml10_url12, ml10_url13, ml10_url14
    return render_template(
        'ml_ch10/ml_ch10_5.html', ml10_url12=ml10_url12, ml10_url13=ml10_url13, ml10_url14=ml10_url14
    )

@app.route('/ml_ch10_6')
def ml_ch10_6():
    from ml_ch10.ml_ch10_6 import ml10_url15, ml10_url16, ml10_url17, ml10_url18, \
        ml10_plot_url8, ml10_plot_url9, ml10_plot_url10, ml10_plot_url11, ml10_plot_url12
    return render_template(
        'ml_ch10/ml_ch10_6.html', ml10_url15=ml10_url15, ml10_url16=ml10_url16, ml10_url17=ml10_url17, ml10_url18=ml10_url18,
        ml10_plot_url8=ml10_plot_url8, ml10_plot_url9=ml10_plot_url9, ml10_plot_url10=ml10_plot_url10,
        ml10_plot_url11=ml10_plot_url11, ml10_plot_url12=ml10_plot_url12
    )


#
#
#

@app.route('/ml_ch11')
def ml_ch11():
    from ml_ch11.ml_ch11 import ml11_url1, ml11_url2, ml11_url3, \
        ml11_plot_url1, ml11_plot_url2, ml11_plot_url3, ml11_plot_url4, ml11_plot_url5, ml11_plot_url6, ml11_plot_url7, ml11_plot_url8, ml11_plot_url9, ml11_plot_url10, ml11_plot_url11
    return render_template(
        'ml_ch11/ml_ch11.html', ml11_url1=ml11_url1, ml11_url2=ml11_url2, ml11_url3=ml11_url3,
        ml11_plot_url1=ml11_plot_url1, ml11_plot_url2=ml11_plot_url2, ml11_plot_url3=ml11_plot_url3, ml11_plot_url4=ml11_plot_url4, ml11_plot_url5=ml11_plot_url5,
        ml11_plot_url6=ml11_plot_url6, ml11_plot_url7=ml11_plot_url7, ml11_plot_url8=ml11_plot_url8, ml11_plot_url9=ml11_plot_url9, ml11_plot_url10=ml11_plot_url10,
        ml11_plot_url11=ml11_plot_url11
    )    

@app.route('/ml_ch11_1')
def ml_ch11_1():
    from ml_ch11.ml_ch11_1 import ml11_url1, ml11_plot_url1, ml11_plot_url2, ml11_plot_url3, ml11_plot_url4, ml11_plot_url5, ml11_plot_url6
    return render_template(
        'ml_ch11/ml_ch11_1.html', ml11_url1=ml11_url1, ml11_plot_url1=ml11_plot_url1, ml11_plot_url2=ml11_plot_url2, ml11_plot_url3=ml11_plot_url3, ml11_plot_url4=ml11_plot_url4, 
        ml11_plot_url5=ml11_plot_url5, ml11_plot_url6=ml11_plot_url6
    )    

@app.route('/ml_ch11_2')
def ml_ch11_2():
    from ml_ch11.ml_ch11_2 import ml11_url2, ml11_url3, \
        ml11_plot_url7, ml11_plot_url8
    return render_template(
        'ml_ch11/ml_ch11_2.html', ml11_url2=ml11_url2, ml11_url3=ml11_url3,
        ml11_plot_url7=ml11_plot_url7, ml11_plot_url8=ml11_plot_url8
    )    

@app.route('/ml_ch11_3')
def ml_ch11_3():
    from ml_ch11.ml_ch11_3 import ml11_plot_url9, ml11_plot_url10, ml11_plot_url11
    return render_template(
        'ml_ch11/ml_ch11_3.html', ml11_plot_url9=ml11_plot_url9, ml11_plot_url10=ml11_plot_url10, ml11_plot_url11=ml11_plot_url11
    )    

#
#
#

@app.route('/ml_ch12')
def ml_ch12():
    from ml_ch12.ml_ch12 import ml12_url1, ml12_url2, ml12_url3, ml12_url4, ml12_url5, \
        ml12_plot_url1, ml12_plot_url2, ml12_plot_url3, ml12_plot_url4, ml12_plot_url5
    return render_template(
        'ml_ch12/ml_ch12.html', ml12_url1=ml12_url1, ml12_url2=ml12_url2, ml12_url3=ml12_url3, ml12_url4=ml12_url4, ml12_url5=ml12_url5,
        ml12_plot_url1=ml12_plot_url1, ml12_plot_url2=ml12_plot_url2, ml12_plot_url3=ml12_plot_url3, ml12_plot_url4=ml12_plot_url4, ml12_plot_url5=ml12_plot_url5
    )    

#
#
#

#
#
#

@app.route('/ml_ch14')
def ml_ch14():
    from ml_ch14.ml_ch14 import ml14_url1, \
        ml14_plot_url1, ml14_plot_url2, ml14_plot_url3
    return render_template(
        'ml_ch14/ml_ch14.html', ml14_url1=ml14_url1,
        ml14_plot_url1=ml14_plot_url1, ml14_plot_url2=ml14_plot_url2, ml14_plot_url3=ml14_plot_url3
    )    

#
#
#

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)