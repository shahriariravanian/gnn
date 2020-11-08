include("gnn.jl")
include("model.jl")
include("prepare.jl")
include("train.jl")
include("neural_ode.jl")

const models_dir="models_mixed"

function generate_all(; mix=false)
    if !ispath(models_dir)
        mkdir(models_dir)
    end

    ℳ = TenTusscherModel()

    names = ["longqt", "shortqt", "ito"]
    mods = [[(:g_Kr, 0.5)], [(:g_CaL, 0.5)], [(:g_to, 2.0)]]

    for (name, mod) in zip(names, mods)
        @info "generating $name\n"
        generate(ℳ, name, mod, mix)
    end
end

function generate(ℳ, name, mod, mix; ηs=1f-4:1f-4:2f-3)
    path = model_path(models_dir, name)
    if !ispath(path)
        ℳ1 = modify_model(ℳ, mod)
        ℬ = prepare(ℳ, ℳ1; mix=mix)
        ℬ = train(ℬ)
        save_model(ℬ, path)
    else
        ℬ = gpu(load_model(path))
    end

    for η in ηs
        @info "\ttraining for $η\n"
        path = submodel_path(models_dir, name, η)
        if !ispath(path)
            ℛ = retrain(ℬ, η)
            save_model(ℛ, path)
        end
    end
end
