/*
    Author: By Thomas Kim
    Date: April 9th 2025
    
    MSVC C++ compiler
    cl FFT-Unified.cpp /Fe: m.exe /std:c++latest /EHsc /nologo

    LLVM Clang
    clang++ -std=c++23 FFT-Unified.cpp -o c.exe

    GNU G++
    g++ -std=c++23 FFT-Unified.cpp -o g.exe

    Intel C++ compiler
    icx-cl FFT-Unified.cpp -o i.exe -Qstd=c++23 /EHsc /nologo
*/

#include <iostream>
#include <complex>
#include <cmath>
#include <numbers>
#include <vector>

using cplx = std::complex<double>;
using cplx_numbers = std::vector<cplx>;

using call_counts = std::vector<int>;
using call_ptr_counts = std::vector<int*>;

using real_parts = std::vector<double>;

double adjust_zero(double d)
{
    constexpr auto epsilon = 10.0E-6;
        // std::numeric_limits<double>::epsilon() * 10 * 10 *;

    return std::abs(d) < epsilon ? 0.0 : d; 
}

cplx adjust_zero(const cplx& c)
{
    return { adjust_zero(c.real()), adjust_zero(c.imag() )};
}

auto cplx_to_double(const cplx_numbers& c)
{
    real_parts r( c.size() );

    for(int n =0; n < c.size(); n++)
        r[n] = adjust_zero(c[n].real());

    return r;
}

std::ostream& operator << ( std::ostream& os, const cplx_numbers& c)
{
    if (c.empty())
    {
        os <<"{ }"; return os;
    }
    else
    {
        os <<"{ ";

        for(int i = 0; i < c.size()-1; ++i)
            os << adjust_zero(c[i]) <<", ";

        os << adjust_zero(c.back()) << " }";

        return os;
    }
}

std::ostream& operator << ( std::ostream& os, const call_counts& c)
{
    if (c.empty())
    {
        os <<"{ }"; return os;
    }
    else
    {
        os <<"{ ";

        for(int i = 0; i < c.size()-1; ++i)
            os << c[i] <<", ";

        os << c.back() << " }";

        return os;
    }
}

std::ostream& operator << ( std::ostream& os, const real_parts& c)
{
    if (c.empty())
    {
        os <<"{ }"; return os;
    }
    else
    {
        os <<"{ ";

        for(int i = 0; i < c.size()-1; ++i)
            os << c[i] <<", ";

        os << c.back() << " }";

        return os;
    }
}

int reverse_bits(int x, int numBits)
{
    int reversed = 0; // N = 16, numBits = log2(16) = 4

    // i = 0, 1, 2, 3
    for(int i = 0; i < numBits; ++i)
    {
        if(x & (1 << i))
        {
            reversed |= 1 << (numBits-1 - i); 
        }
    }

    return reversed;
}

/*
    using cplx = std::complex<double>;
    using cplx_numbers = std::vector<cplx>;
*/
void shuffle_coefficients(cplx_numbers& data)
{
    int N = data.size(); // 1, 2, 4, 8, 16, ... 2^n, n = 0, 1, 2, 3

    // N = 16, numBits = 4
    int numBits = log2(N); // N = 16, log2(16) = log2(2^4) = 4 log2(2) = 4;

    for(int x = 0; x < N; ++x)
    {
        int reverse_index = reverse_bits(x, numBits);

        if(reverse_index > x)
        {
            std::swap(data[x], data[reverse_index]);
        }
    }
}

void fft_loop(cplx_numbers& coeffs, bool postive = true)
{
    constexpr auto pi = std::numbers::pi_v<double>;

    int N = coeffs.size();

    cplx_numbers power(N);

    for(int n = 0; n < N/2; ++n)
        power[n]= postive ? std::exp( cplx{ 0.0, 2 * pi * n / N } ) : 
            std::exp( cplx{ 0.0, -2 * pi * n / N } );

    shuffle_coefficients(coeffs);

    for(int len = 2; len <= N; len *= 2 )
    {
        for(int i = 0;  i < N; i += len)
        {
            for(int j = 0; j < len/2; ++j)
            {
                auto left_even = coeffs[i+j];
                auto right_odd  = coeffs[i+j + len/2];
                
                // auto w = postive ? std::exp( cplx{ 0.0, 2 * pi * j / len} ) : 
                //         std::exp( cplx{ 0.0, -2 * pi * j / len} );
                auto w = power[j * (N/len)];

                coeffs[i+j] = left_even + w * right_odd;
                coeffs[len/2 + i+j] = left_even - w * right_odd;
            }
        }
    }
}

enum class scaling_factor { NO_Scaling, FFT_N, IFFT_N, Both_sqrt_N };

template<scaling_factor factor>
auto get_fft_ifft()
{
    if constexpr(factor == scaling_factor::NO_Scaling)
    {
        auto fft = [](cplx_numbers& coeffs)
        {
            fft_loop(coeffs, false); // e^-2pi i /N
        };

        auto ifft = [](cplx_numbers& coeffs)
        {
            fft_loop(coeffs, true); // e^2pi i /N
        };

        return std::tuple{ fft, ifft };
    }
    else if constexpr(factor == scaling_factor::FFT_N)
    {
        auto fft = [](cplx_numbers& coeffs)
        {
            fft_loop(coeffs, false); // e^-2pi i /N

            double N = coeffs.size();

            for(int i = 0; i < coeffs.size(); ++i)
                coeffs[i] /= N;
        };

        auto ifft = [](cplx_numbers& coeffs)
        {
            fft_loop(coeffs, true); // e^2pi i /N
        };

        return std::tuple{ fft, ifft };
    }
    else if constexpr(factor == scaling_factor::IFFT_N)
    {
        auto fft = [](cplx_numbers& coeffs)
        {
            fft_loop(coeffs, false); // e^-2pi i /N
        };

        auto ifft = [](cplx_numbers& coeffs)
        {
            fft_loop(coeffs, true); // e^2pi i /N

            double N = coeffs.size();

            for(int i = 0; i < coeffs.size(); ++i)
                coeffs[i] /= N;
        };

        return std::tuple{ fft, ifft };
    }
    else // factor == scaling_factor::Both_sqrt_N
    {
        auto fft = [](cplx_numbers& coeffs)
        {
            fft_loop(coeffs, false); // e^-2pi i /N

            double N = std::sqrt(coeffs.size());

            for(int i = 0; i < coeffs.size(); ++i)
                coeffs[i] /= N;

        };

        auto ifft = [](cplx_numbers& coeffs)
        {
            fft_loop(coeffs, true); // e^2pi i /N

            double N = std::sqrt(coeffs.size());

            for(int i = 0; i < coeffs.size(); ++i)
                coeffs[i] /= N;
        };

        return std::tuple{ fft, ifft };
    }
}

void test_Laurent_vs_Fourier_with_FFT_IFFT()
{
    using enum scaling_factor;

    auto epsilon = std::numeric_limits<double>::epsilon();

    auto [fft, ifft] = get_fft_ifft<IFFT_N>();

    constexpr auto pi = std::numbers::pi_v<double>;

    // periodic cos function
    auto f_cos = [](auto x)
    {
        return std::cos(x);
    };

    // function with singularity at x = 0, center c = 0.0
    // we assume we know the center of the function,
    // but we do not know the coefficients:4.0, 2.0, 1.0, 5.0, 3.0
    auto f_sgt_zero = [](auto x)
    {
        return 4.0 / (x*x) + 2.0 / x + 1.0 + 5.0 * x + 3.0 * x * x;
    };

    auto f_poly = [](auto x)
    {
        return 1.0 + 2.0 * x + 3.0 * x * x + 4.0 * x * x * x;
    };

    // function with singularity at x = 1, center c = 1.0
    // we assume we know the center of the function.
    // but we do not know the coefficient: 7.0, 5.0, 2.0, 3.0, 6.0
    // this function is hard-coded, that is, my algorithm does not
    // know the coefficients
    auto f_sgt_one = [](auto x)
    {
        return 7.0 / ( (x - 1.0)* (x - 1.0) ) + 5.0 / (x - 1.0)
               + 2.0 + 3.0 * (x - 1.0) + 6.0 * (x - 1.0) * (x - 1.0);
    };

    auto f_taylor = [epsilon](auto x)
    {
        return 1.0/(1.0 - x/2.0);
    };

    auto f_sampling = [](auto x)
    {
        return 1.0 / (1.0 - x);
    };

    auto f_sampling_plus = [](auto x)
    {
        return 1.0 / (1.0 + x);
    };

    int N = 16; // N = 2^n = 1, 2, 4, 8, 16, ..., n = 0, 1, 2, 3, 
    
    auto w = [N, pi](auto n)
    {
        // N-th roots of unity, e^(2 pi i k / N)
        return std::exp( cplx{ 0.0, 2 * pi * n / N });
    };

    auto rw = [N, pi](auto k)
    {
        // z_k = {e^(2 pi / 4 i + 2 pi i k)}^(1/N)
        // z_k = {e^{(2pi)(1/4 + k)}}^(1/N)
        return std::exp( cplx{ 0.0,  2 * pi / (4 * N) }) * 
            std::exp( cplx{ 0.0,     2 * pi * k / N });
    };

    auto rrw = [N, pi](auto k)
    {
        return std::exp( cplx{ 0.0,  2 * pi / (4 * N) }) * std::pow(0.99999999, k/(double)N) * 
            std::exp( cplx{ 0.0,     2 * pi * k / N });
    };
    
    auto wr = [N, pi](auto k)
    {
        return std::exp( cplx{ 0.0,  2 * pi / (4 * N) }) * std::pow(0.99999999, k/(double)N) * 
            std::exp( cplx{ 0.0,     2 * pi * k / N });
    };

    auto f_pui_taylor = [](auto z)
    {
        return 1.0 / (1.0 + std::sqrt(z/2.0) );
    };

    auto f_laurent = [f_taylor](auto z)
    {
        auto x = 2.0 - z;
        return f_taylor(x);
    };

    auto g_stg_one = [f_sgt_one](auto z)
    {
        /*
            f(x) = 7.0 / ( (x - 1.0)* (x - 1.0) ) + 5.0 / (x - 1.0)
                   + 2.0 + 3.0 * (x - 1.0) + 6.0 * (x - 1.0) * (x - 1.0);

            g(x + 1.0) = 7.0 / (z*z) + 5.0 / z
                        + 2.0 + 3.0 * z + 6.0 * z * z;
            
            Step 1. we transform function f(x) to function g(x).

            we let x - 1.0 = z, then x = z + 1.0
        */

        auto x = z + 1.0; // 1.0 is the center

        return f_sgt_one(x);
    };

    std::vector<cplx> fn_poly(N);
    std::vector<cplx> fn_taylor(N);
    std::vector<cplx> an_taylor(N);
    std::vector<cplx> fn_sampling(N);
    std::vector<cplx> an_sampling(N);

    std::vector<cplx> fn_sampling_plus(N);
    std::vector<cplx> an_sampling_plus(N);

    std::vector<cplx> fn_laurent(N);
    std::vector<cplx> fn_cos(N);
    
    std::vector<cplx> fn_cos_sampling(N);
    std::vector<cplx> an_cos_sampling(N);

    std::vector<cplx> fn_sgt_zero(N);
    std::vector<cplx> fn_sgt_one(N);

    // Sampling for Taylor-Laurent series: c + e^(2 pi i k /N )
    // The reason we have divide the function values with N
    // is to match the expected function values format of FFT,
    // or the generated results of IFFT_N
    // In the sampling loop
    for(int n = 0; n < N; ++n)
    {
     
        fn_cos[n] = f_cos( w(n) ) / (double)N;
        fn_taylor[n] = f_taylor( w(n) ) / (double)N;

        fn_laurent[n] = w(n) * f_laurent( w(n) ) /(double)N;

        fn_poly[n] = f_poly( rrw(n) ) / (double)N;

        fn_cos_sampling[n]      = f_cos(rrw(n))     /(double)(N);

        fn_sampling[n]      = f_sampling(rrw(n))     /(double)(N/2);
        fn_sampling_plus[n] = f_sampling_plus(rrw(n))/(double)(N/2);

        // In this case, we have to rotate 2 places due to x^-2,
        // because 4.0 / (x*x) + 2.0 / x + 1.0 + 5.0 * x + 3.0 * x * x,
        // is Laurent series. We transform it to Taylor series format LATER.
        
        // fn_sgt_zero[n] = w(n) * w(n) * f_sgt_zero( w(n) ) / (double)N;
        // or we can rotate 2 places in the following case 
        fn_sgt_zero[n] = f_sgt_zero( w(n) ) / (double)N; 

        // we can rotate 2 places due to z^-2,
        // because 7.0 / (z*z) + 5.0 / z + 2.0 + 3.0 * z + 6.0 * z * z
        // has negative power 2, we can rotate Later as in the above case,
        // but if we multiply w(n), twiddle factor, we are actually rotating once.
        // so, we can rotate twice by multiplying w(n), (one property of N-th roots of unity)
        // which amounts to Transforming Laurent series to Taylor series, the end result
        // looks like 7.0 + 5.0 * z + 2.0 * z^2 + 3.0 * z^3 + 6.0 * z^4,
        // which is Taylor series.
        fn_sgt_one[n] =  w(n) * w(n) * g_stg_one( w(n) ) / (double)N; 
    }


    // Input: function values matching the format of FFT,
    // Ouput: coefficients of the function.
    // The Input and Output can be reversed.

    // std::cout << "Taylor coefficients Sampling of 1/(1-x/2): " 
    //         << fn_taylor << std::endl << std::endl;

    std::cout << "Taylor coefficients Sampling of 1/(1-x): " 
            << fn_sampling << std::endl << std::endl;

    fft(fn_cos); fft(fn_cos_sampling);
    fft(fn_sgt_zero); fft(fn_sgt_one); 
    fft(fn_taylor); fft(fn_laurent); fft(fn_sampling);
    fft(fn_sampling_plus);

    for(int n = 0; n < N; ++n)
    {
        an_cos_sampling[n] = fn_cos_sampling[n] /
        ( std::pow(0.99999999, n/(double)N) * std::exp( cplx{ 0.0, 2 * pi * n / (4 * N) }) );

        an_sampling[n] = fn_sampling[n] /
            ( std::pow(0.99999999, n/(double)N) * std::exp( cplx{ 0.0, 2 * pi * n / (4 * N) }) );

        an_sampling_plus[n] = fn_sampling_plus[n] /
        ( std::pow(0.99999999, n/(double)N) * std::exp( cplx{ 0.0, 2 * pi * n / (4 * N) }) );
    }

    std::cout << "Taylor coefficients an_cos_sampling of cos(x): ";
    std::cout << cplx_to_double(an_cos_sampling) << std::endl << std::endl;

    std::cout << "Taylor coefficients an_sampling of 1/(1-x): ";
    std::cout << cplx_to_double(an_sampling) << std::endl << std::endl;

    std::cout << "Taylor coefficients an_sampling_plus of 1/(1+x): ";
    std::cout << cplx_to_double(an_sampling_plus) << std::endl << std::endl;


    // we take only real values from the fft returned values
    std::cout << "Taylor coefficients of cos(x): " 
            << cplx_to_double(fn_cos) << std::endl << std::endl;

    // we take only real values from the fft returned values
    std::cout << "Laurent coefficients of 1/(1 - x/2): " 
            << cplx_to_double(fn_laurent) << std::endl << std::endl;

            
    for(int k = 0; k < N; ++k)
        an_taylor[k] = fn_taylor[k] / std::pow(2.0, k / (double)N);

    std::cout << "Taylor coefficients of 1/(1 - x/2.0 ): " 
            << cplx_to_double(an_taylor) << std::endl << std::endl;
    
    std::cout << "Laurent coefficients of f_sgt_zero(x): ";

    // we have to rotate 2 places
    // fn_sgt_zero.rbegin() + 2 accounts for w(n) * w(n)
    std::rotate(fn_sgt_zero.rbegin(), fn_sgt_zero.rbegin() + 2, fn_sgt_zero.rend());

    std::cout << cplx_to_double(fn_sgt_zero) << std::endl << std::endl;

    
    // because we multiplied w(n) * w(n) when sampling
    // we don't need to rotate 2 places
    std::cout << "Laurent coefficients of f_sgt_one(x): "
         << cplx_to_double(fn_sgt_one) << std::endl << std::endl;

    // Sample for Fourier-Series (or Discrete Fourier Transform)
    // Fundamentally Taylor-Laurent series and Fourier series
    // are the Same. It is Thomas Kim who saw this for the first time in Human History.

    // periodic 1.0 + 2 * cos(2x) + 3 * sin(4x) function
    auto f_cos_plus = [](auto x)
    {
        return 1.0 + 2.0 * std::cos(2.0 * x) + 3.0 * std::sin(4.0 * x);
    };

    // Sampling for Fourier series
    // we are sampling f(x) = 1.0 + 2 * cos(2x) + 3 * sin(4x)
    // this is for Fourier series
    std::vector<cplx> fn_cos_fourier(N);

    for(int n = 0; n < N; ++n)
    {
        // sampling at equidistant N points over the whole period or interval
        fn_cos_fourier[n] = f_cos_plus( 2 * pi * n / N);
    }

    fft(fn_cos_fourier);

    std::vector<double> an_cos, bn_cos;
       
    // Bn = -Im(an) * (2/N)

    // A0 = Re(a0) / N,
    an_cos.emplace_back( adjust_zero (fn_cos_fourier[0].real() / N ) );

    for(int n = 1; n <= N/2; ++n)
    {
        // An = Re(an) * (2/N)
        an_cos.emplace_back( adjust_zero(fn_cos_fourier[n].real() * (2.0/N) ) );

        // Bn = Im(an) * (-2/N)
        bn_cos.emplace_back( adjust_zero(fn_cos_fourier[n].imag() * (-2.0/N) ) );
    }

    std::cout <<"Fourier Coefficients of 1.0 + 2 * cos(2x) + 3 * sin(4x)\n";
    std::cout <<"an = " << an_cos << std::endl; 
    std::cout <<"bn = " << bn_cos << std::endl << std::endl; 
}

int main()
{
    test_Laurent_vs_Fourier_with_FFT_IFFT();
}

𝑧