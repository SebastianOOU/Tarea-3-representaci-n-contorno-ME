# Serie de Fourier compleja de UN recorrido cerrado que pasa por todo el dibujo del gato
# (contorno, ojos, nariz, bigotes y pelo). Todo en Julia, sin Python ni archivos intermedios.
#
# Entrada : gato.png (dibujo de líneas negras; puede tener fondo transparente)
# Salida  : carpeta salida_julia/
#
# Pasos:
#   1) Imagen -> tinta (se compone sobre fondo blanco) -> esqueleto de 1 píxel (Zhang-Suen)
#   2) Esqueleto -> grafo de píxeles; se podan ramitas muy cortas y motas
#   3) Las piezas sueltas (ojos, bigotes...) se unen con puentes rectos mínimos
#   4) Cartero chino: se repiten los tramos justos para que todo nodo tenga grado par
#   5) Circuito euleriano (Hierholzer) = recorrido cerrado que usa cada tramo
#   6) Parametrización por longitud de arco, c_n = fft(z)/N, reconstrucciones, errores, animación
#
# Paquetes (una sola vez):  using Pkg; Pkg.add(["Images","FFTW","Plots","DelimitedFiles"])
using Images, FFTW, Plots, DelimitedFiles, Statistics

cd(@__DIR__)
mkpath("salida_julia")

const ARCHIVO = "gato.png"
const N  = 32768                                    # muestras del recorrido (M nunca puede superar N)
const Ms = [10, 25, 50, 100, 200, 500, 1000, 2000, 4000, 8192]

# ======================================================================
# Utilidades
# ======================================================================

# Imagen -> máscara de tinta (BitMatrix). Se compone sobre blanco: un PNG con fondo
# transparente tiene píxeles negros "invisibles" que, sin esto, se confundirían con la tinta.
function cargar_tinta(archivo)
    img = RGBA.(load(archivo))
    gris = map(img) do p
        a = Float64(alpha(p))
        g = 0.299 * Float64(red(p)) + 0.587 * Float64(green(p)) + 0.114 * Float64(blue(p))
        a * g + (1 - a)
    end
    return gris .< 0.5
end

# Adelgazamiento de Zhang-Suen: deja líneas de 1 píxel de grosor
function zhang_suen(tinta)
    img = copy(tinta)
    nr, nc = size(img)
    cambio = true
    while cambio
        cambio = false
        for paso in 1:2
            borrar = Tuple{Int,Int}[]
            for j in 2:nc-1, i in 2:nr-1
                img[i, j] || continue
                p2 = img[i-1, j];   p3 = img[i-1, j+1]; p4 = img[i, j+1];   p5 = img[i+1, j+1]
                p6 = img[i+1, j];   p7 = img[i+1, j-1]; p8 = img[i, j-1];   p9 = img[i-1, j-1]
                B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                (2 <= B <= 6) || continue
                A = (!p2 & p3) + (!p3 & p4) + (!p4 & p5) + (!p5 & p6) +
                    (!p6 & p7) + (!p7 & p8) + (!p8 & p9) + (!p9 & p2)
                A == 1 || continue
                ok = paso == 1 ? (!(p2 & p4 & p6) && !(p4 & p6 & p8)) :
                                 (!(p2 & p4 & p8) && !(p2 & p6 & p8))
                ok || continue
                push!(borrar, (i, j))
            end
            for (i, j) in borrar
                img[i, j] = false
            end
            isempty(borrar) || (cambio = true)
        end
    end
    return img
end

# Montículo binario mínimo (para Dijkstra), sin depender de otros paquetes
mutable struct Monticulo
    d::Vector{Float64}
    v::Vector{Int}
end
Monticulo() = Monticulo(Float64[], Int[])

function meter!(h::Monticulo, d, v)
    push!(h.d, d); push!(h.v, v)
    i = length(h.d)
    while i > 1
        p = i ÷ 2
        h.d[p] <= h.d[i] && break
        h.d[p], h.d[i] = h.d[i], h.d[p]
        h.v[p], h.v[i] = h.v[i], h.v[p]
        i = p
    end
end

function sacar!(h::Monticulo)
    d0, v0 = h.d[1], h.v[1]
    h.d[1] = h.d[end]; h.v[1] = h.v[end]
    pop!(h.d); pop!(h.v)
    n = length(h.d)
    i = 1
    while true
        l = 2i; r = 2i + 1; m = i
        (l <= n && h.d[l] < h.d[m]) && (m = l)
        (r <= n && h.d[r] < h.d[m]) && (m = r)
        m == i && break
        h.d[m], h.d[i] = h.d[i], h.d[m]
        h.v[m], h.v[i] = h.v[i], h.v[m]
        i = m
    end
    return d0, v0
end

# ======================================================================
# Esqueleto -> recorrido cerrado (cartero chino + circuito euleriano)
# ======================================================================
function ruta_cerrada(skel; spur=9, min_pieza=20)
    nr, nc = size(skel)

    # ---- nodos y grafo de píxeles (sin triángulos de escalera) ----
    idx = zeros(Int, nr, nc)
    ys = Int[]; xs = Int[]
    for j in 1:nc, i in 1:nr
        if skel[i, j]
            push!(ys, i); push!(xs, j)
            idx[i, j] = length(ys)
        end
    end
    n = length(ys)
    X = Float64.(xs); Y = Float64.(ys)
    hay(i, j) = 1 <= i <= nr && 1 <= j <= nc && skel[i, j]
    adj = [Int[] for _ in 1:n]
    conecta!(a, b) = (push!(adj[a], b); push!(adj[b], a))
    for k in 1:n
        i, j = ys[k], xs[k]
        hay(i, j+1) && conecta!(k, idx[i, j+1])
        hay(i+1, j) && conecta!(k, idx[i+1, j])
        hay(i+1, j+1) && !hay(i, j+1) && !hay(i+1, j) && conecta!(k, idx[i+1, j+1])
        hay(i+1, j-1) && !hay(i, j-1) && !hay(i+1, j) && conecta!(k, idx[i+1, j-1])
    end
    alive = trues(n)
    grado(v) = count(u -> alive[u], adj[v])
    vecinos(v) = [u for u in adj[v] if alive[u]]
    println("Esqueleto: ", n, " píxeles")

    # ---- poda de ramitas terminales más cortas que `spur` ----
    function podar!(L)
        for hoja in 1:n
            (alive[hoja] && grado(hoja) == 1) || continue
            cam = [hoja]; prev = 0; cur = hoja
            while true
                nb = [u for u in vecinos(cur) if u != prev]
                length(nb) == 1 || break
                prev, cur = cur, nb[1]
                grado(cur) == 2 || break
                push!(cam, cur)
            end
            if grado(cur) >= 3 && length(cam) < L
                for v in cam
                    alive[v] = false
                end
            end
        end
    end
    podar!(spur); podar!(spur)
    for v in 1:n
        (alive[v] && grado(v) == 0) && (alive[v] = false)
    end

    # ---- componentes conexas ----
    function etiquetar()
        comp = zeros(Int, n)
        total_c = 0
        for s in 1:n
            (alive[s] && comp[s] == 0) || continue
            total_c += 1
            comp[s] = total_c
            pila = [s]
            while !isempty(pila)
                w = pop!(pila)
                for z in adj[w]
                    if alive[z] && comp[z] == 0
                        comp[z] = total_c
                        push!(pila, z)
                    end
                end
            end
        end
        lista = [Int[] for _ in 1:total_c]
        for w in 1:n
            alive[w] && push!(lista[comp[w]], w)
        end
        return total_c, lista
    end
    ncomp, miembros = etiquetar()
    for m in miembros                              # motas: piezas de menos de `min_pieza` píxeles
        if length(m) < min_pieza
            for v in m
                alive[v] = false
            end
        end
    end
    ncomp, miembros = etiquetar()
    println("Piezas del dibujo: ", ncomp, " (tamaños: ", sort(length.(miembros), rev=true), ")")

    # ---- puentes rectos entre piezas (árbol de expansión mínima, Kruskal) ----
    function par_cercano(mi, mj)
        mejor = Inf; bu = 0; bv = 0
        for u in mi, v in mj
            d = (X[u] - X[v])^2 + (Y[u] - Y[v])^2
            if d < mejor
                mejor = d; bu = u; bv = v
            end
        end
        return sqrt(mejor), bu, bv
    end
    cand = Tuple{Float64,Int,Int,Int,Int}[]
    for i in 1:ncomp, j in i+1:ncomp
        d, u, v = par_cercano(miembros[i], miembros[j])
        push!(cand, (d, u, v, i, j))
    end
    sort!(cand)
    padre = collect(1:ncomp)
    function raiz(a)
        while padre[a] != a
            padre[a] = padre[padre[a]]
            a = padre[a]
        end
        return a
    end
    puentes = Dict{Tuple{Int,Int},Float64}()
    for (d, u, v, i, j) in cand
        ri, rj = raiz(i), raiz(j)
        if ri != rj
            padre[ri] = rj
            push!(adj[u], v); push!(adj[v], u)
            puentes[(min(u, v), max(u, v))] = d
        end
    end
    println("Puentes: ", length(puentes), " | longitud: ", round(sum(values(puentes)), digits=1), " px")

    function peso(u, v)
        w = get(puentes, (min(u, v), max(u, v)), -1.0)
        return w >= 0 ? w : hypot(X[u] - X[v], Y[u] - Y[v])
    end

    # ---- cartero chino: Dijkstra desde cada nodo de grado impar ----
    impares = [v for v in 1:n if alive[v] && isodd(grado(v))]
    P = length(impares)
    println("Nodos de grado impar: ", P)
    function dijkstra(src)
        dist = fill(Inf, n); par = zeros(Int, n)
        dist[src] = 0.0
        h = Monticulo()
        meter!(h, 0.0, src)
        while !isempty(h.d)
            d, v = sacar!(h)
            d > dist[v] && continue
            for u in adj[v]
                alive[u] || continue
                nd = d + peso(v, u)
                if nd < dist[u]
                    dist[u] = nd; par[u] = v
                    meter!(h, nd, u)
                end
            end
        end
        return dist, par
    end
    Dm = zeros(P, P)
    pars = Vector{Vector{Int}}(undef, P)
    for a in 1:P
        dist, par = dijkstra(impares[a])
        pars[a] = par
        for b in 1:P
            Dm[a, b] = dist[impares[b]]
        end
    end

    # emparejamiento de los nodos impares: voraz + mejora 2-opt
    pares = Tuple{Float64,Int,Int}[(Dm[i, j], i, j) for i in 1:P for j in i+1:P]
    sort!(pares)
    libre = trues(P)
    empar = Vector{Vector{Int}}()
    for (d, i, j) in pares
        if libre[i] && libre[j]
            libre[i] = false; libre[j] = false
            push!(empar, [i, j])
        end
    end
    costo0 = sum(Dm[e[1], e[2]] for e in empar)
    mejoro = true
    while mejoro
        mejoro = false
        for p in 1:length(empar), q in p+1:length(empar)
            a, b = empar[p]; c, d = empar[q]
            act = Dm[a, b] + Dm[c, d]
            o1 = Dm[a, c] + Dm[b, d]
            o2 = Dm[a, d] + Dm[b, c]
            if o1 < act - 1e-9 && o1 <= o2
                empar[p] = [a, c]; empar[q] = [b, d]; mejoro = true
            elseif o2 < act - 1e-9
                empar[p] = [a, d]; empar[q] = [b, c]; mejoro = true
            end
        end
    end
    costo1 = sum(Dm[e[1], e[2]] for e in empar)
    println("Emparejamiento: voraz ", round(costo0), " -> mejorado ", round(costo1), " px")

    # ---- multigrafo: aristas originales + tramos repetidos ----
    au = Int[]; av = Int[]; abr = Bool[]
    inc = [Int[] for _ in 1:n]
    function add_arista!(u, v)
        push!(au, u); push!(av, v)
        push!(abr, haskey(puentes, (min(u, v), max(u, v))))
        e = length(au)
        push!(inc[u], e); push!(inc[v], e)
    end
    for u in 1:n
        alive[u] || continue
        for v in vecinos(u)
            u < v && add_arista!(u, v)
        end
    end
    n_base = length(au)
    total = sum(peso(au[e], av[e]) for e in 1:n_base)
    extra = 0.0
    for (i, j) in empar
        cur = impares[j]
        while cur != impares[i]
            p = pars[i][cur]
            add_arista!(p, cur)
            extra += peso(p, cur)
            cur = p
        end
    end
    println("Longitud del esqueleto: ", round(total), " px | tramos repetidos: ", round(extra),
            " px (", round(100 * extra / total, digits=1), " %)")
    @assert all(iseven(length(inc[v])) for v in 1:n) "quedó algún nodo de grado impar"

    # ---- circuito euleriano (Hierholzer, iterativo) ----
    inicio = argmin([alive[v] ? X[v] : Inf for v in 1:n])
    usada = falses(length(au))
    ptr = ones(Int, n)
    pv = [inicio]; pe = [0]
    cv = Int[]; ce = Int[]
    while !isempty(pv)
        v = pv[end]
        while ptr[v] <= length(inc[v]) && usada[inc[v][ptr[v]]]
            ptr[v] += 1
        end
        if ptr[v] > length(inc[v])
            pop!(pv); e = pop!(pe)
            push!(cv, v); push!(ce, e)
        else
            e = inc[v][ptr[v]]
            usada[e] = true
            push!(pv, au[e] == v ? av[e] : au[e])
            push!(pe, e)
        end
    end
    @assert all(usada) "no se recorrieron todas las aristas"
    @assert cv[1] == cv[end] == inicio "el recorrido no cierra"

    # ---- coordenadas del recorrido: z = x + i·y (con y hacia arriba) ----
    pts = ComplexF64[complex(X[cv[1]], -Y[cv[1]])]
    for k in 1:length(cv)-1
        e = ce[k]; u = cv[k]; v = cv[k+1]
        if abr[e]
            m = max(2, floor(Int, hypot(X[v] - X[u], Y[v] - Y[u])))
            for t in range(0, 1, length=m + 1)[2:end]
                push!(pts, complex(X[u] + t * (X[v] - X[u]), -(Y[u] + t * (Y[v] - Y[u]))))
            end
        else
            push!(pts, complex(X[v], -Y[v]))
        end
    end
    return pts
end

# ======================================================================
# Fourier
# ======================================================================

# Parametrización por longitud de arco -> N muestras uniformes en t ∈ [0,1)
function remuestrear(pts, N)
    s = [0.0; cumsum(abs.(diff(pts)))]
    s ./= s[end]
    z = Vector{ComplexF64}(undef, N)
    for i in 1:N
        t  = (i - 1) / N
        j  = clamp(searchsortedlast(s, t), 1, length(s) - 1)
        ds = s[j+1] - s[j]
        λ  = ds > 0 ? (t - s[j]) / ds : 0.0
        z[i] = pts[j] + λ * (pts[j+1] - pts[j])
    end
    return z
end

# Reconstrucción con M términos simétricos n = -M/2, ..., M/2-1 (ifft del espectro recortado)
function recon(c, M)
    n  = length(c)
    cm = zeros(ComplexF64, n)
    for m in -(M÷2):(M÷2 - 1)
        cm[mod(m, n) + 1] = c[mod(m, n) + 1]
    end
    return ifft(cm) .* n
end

# Error relativo: % del radio RMS de la figura
err_rel(z, zr) = 100 * sqrt(mean(abs2, zr .- z)) / sqrt(mean(abs2, z .- mean(z)))

function animar(z, c, M, archivo; frames=240)
    n  = length(c)
    ms = sort(collect(-(M÷2):(M÷2 - 1)), by = m -> (abs(m), -m))    # 0, 1, -1, 2, -2, ...
    zM = recon(c, M)
    xl = extrema(real.(z)); yl = extrema(imag.(z))
    fondo = z[1:8:end]
    anim = @animate for f in 0:frames-1
        t = f / frames
        pos = ComplexF64[0]
        acum = 0.0im
        for m in ms
            acum += c[mod(m, n) + 1] * cis(2π * m * t)
            push!(pos, acum)
        end
        idx = max(2, round(Int, t * n) + 1)
        p = plot(real.(fondo), imag.(fondo), color=:gray, alpha=0.25, lw=0.5, label="",
                 aspect_ratio=:equal, xlims=xl, ylims=yl, framestyle=:none, size=(700, 900))
        plot!(p, real.(pos), imag.(pos), color=:steelblue, alpha=0.5, lw=0.5, label="")
        plot!(p, real.(zM[1:4:idx]), imag.(zM[1:4:idx]), color=:red, lw=1, label="")
        scatter!(p, [real(acum)], [imag(acum)], color=:red, ms=3, label="")
    end
    gif(anim, archivo, fps=30)
end

# ======================================================================
# Programa principal
# ======================================================================
function main()
    println("Leyendo ", ARCHIVO, " ...")
    tinta = cargar_tinta(ARCHIVO)
    println("Fracción de tinta: ", round(100 * sum(tinta) / length(tinta), digits=2), " %")
    skel = zhang_suen(tinta)
    pts = ruta_cerrada(skel)
    println("Puntos del recorrido: ", length(pts),
            " | longitud total (px): ", round(sum(abs.(diff(pts))), digits=1),
            " | cierra: ", isapprox(pts[1], pts[end]))
    writedlm("salida_julia/ruta_cerrada.csv", [real.(pts) imag.(pts)], ',')

    z  = remuestrear(pts, N)
    c  = fft(z) ./ N  # c_n ≈ (1/N) Σ z_k e^{-2πi n k/N}
    cf(n) = c[mod(n, N) + 1]

    # Los 100 términos pedidos en el enunciado: n = -50, ..., 49
    ns = -50:49
    cn = [cf(n) for n in ns]
    writedlm("salida_julia/coeficientes_100.csv",
             vcat(["n" "Re" "Im" "abs" "arg"],
                  Any[collect(ns) real.(cn) imag.(cn) abs.(cn) angle.(cn)]), ',')
    println("\nTop 10 coeficientes (de los 100) por magnitud:")
    for j in sortperm(abs.(cn), rev=true)[1:10]
        println("  n = ", lpad(ns[j], 4), "   |c_n| = ", round(abs(cn[j]), digits=2),
                "   arg = ", round(angle(cn[j]), digits=3))
    end

    errores = Float64[]
    println("\nError de reconstrucción:")
    for M in Ms
        e = err_rel(z, recon(c, M))
        push!(errores, e)
        println("  M = ", lpad(M, 4), "  error = ", round(e, digits=2), " %")
    end
    writedlm("salida_julia/tabla_errores.csv", vcat(["M" "error_pct"], Any[Ms errores]), ',')

    plot(Ms, errores, marker=:circle, lw=2, xscale=:log10, yscale=:log10, legend=false,
         xlabel="Número de términos M", ylabel="Error relativo (%)",
         title="Error de reconstrucción del recorrido cerrado")
    savefig("salida_julia/error_vs_terminos.png")

    ps = []
    for M in [100, 500, 1000, 2000, 4000, 8192]
        zr = recon(c, M)
        p  = plot(real.(z), imag.(z), lw=0.5, color=:black, label="", aspect_ratio=:equal,
                  framestyle=:none, title="$M términos (error $(round(err_rel(z, zr), digits=1)) %)")
        plot!(p, real.(zr), imag.(zr), lw=0.6, color=:red, label="")
        push!(ps, p)
    end
    plot(ps..., layout=(2, 3), size=(1500, 1100))
    savefig("salida_julia/comparacion_terminos.png")

    plot(1:2000, [abs(cf(n)) for n in 1:2000], xscale=:log10, yscale=:log10, lw=1.2, legend=false,
         xlabel="n", ylabel="|c_n|", title="Espectro de amplitud")
    savefig("salida_julia/espectro.png")

    animar(z, c, 100, "salida_julia/epiciclos_100.gif")
    println("\nListo. Revisen la carpeta salida_julia/")
end

main()