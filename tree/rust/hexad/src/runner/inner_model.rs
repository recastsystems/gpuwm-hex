// Simplified inner model, boundary layer (0.6.2). Included in schemes.rs to share its row binding.
// No effect unless AD_IM_BL=fixedk: every other run, default included, takes the code paths it took before, word for
// word. With it, each MYNN-EDMF launch's tangent and adjoint is cuda/inner_model.cu's linear diffusion with the
// trajectory's own diffusivities held, in place of the scheme's own derivative (taped or tape-free). The diffusivities
// are MYNN's own K = el sqrt(qke) S (S = sh3d for heat and vapour, sm3d for momentum) from the state the scheme carries
// between calls (el_pbl, qke, sh3d, sm3d as they stand before the launch: the last call's), not exch_h / exch_m, which
// the scheme only writes (THE BREAKAGE THIS PREVENTS: before the launch they are 0 in every column and level, read on
// the 12 km example, so the first build's diffusion mixed nothing and its 15-minute theta tangent was 2.3 times the
// real change). AD_IM_BL=held (reduced form without the boundary layer): the same launch with no mixing and no surface
// exchange, so the boundary layer's tendencies are held at the trajectory's (tangent 0). No tape, no count pass, no arena: the launch
// reads the forward's pre-state words the bind already copied. AD_IM_SFC = all (default) | momentum | none picks the
// held surface exchange; AD_IM_COND = TAU (seconds) adds the second form's smooth large-scale condensation. The door's --inner-model simple sets this word together with the other holds of the mode.
const IM_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inner_model.fatbin"));
const IM_MAX_OUT: usize = 96;
const IM_NK_MAX: i64 = 160;
/// The boundary layer's arguments the scheme reads and never writes: their tangent passes through the launch unchanged.
const IM_BL_INPUTS: [&str; 39] = ["dx", "xland", "ps", "ts", "qsfc", "ust", "ch", "hfx", "qfx", "wspd", "znt", "uoce",
    "voce", "dz", "u", "v", "w", "th", "tt", "p", "exner", "rho", "qv", "qc", "qi", "qs", "nc", "ni", "nifa", "nwfa",
    "nbca", "qoz", "rthraten", "pattern_spp", "frp_mean", "emis_ant_no", "chem3d", "settle3d", "vd3d"];
/// The four tendencies the linear diffusion writes (theta, vapour, u, v).
const IM_BL_TEND: [&str; 4] = ["rthblten", "rqvblten", "rublten", "rvblten"];

pub(crate) fn im_bl_on() -> bool { matches!(env_var("AD_IM_BL").as_deref(), Some("fixedk") | Some("held")) }

static IM_FN: std::sync::OnceLock<Result<usize, String>> = std::sync::OnceLock::new();
unsafe fn im_function(cuda: &Cuda) -> Result<*mut c_void, String> {
    IM_FN.get_or_init(|| {
        let mut handle = ptr::null_mut();
        #[allow(unused_unsafe)]
        let status = unsafe { (cuda.module_load)(&mut handle, IM_IMAGE.as_ptr().cast()) };
        if status != 0 {
            #[allow(unused_unsafe)]
            let class = unsafe { cuda.card_class() };
            return Err(crate::cuda::load_refusal("cuModuleLoadData inner_model (AD_IM_BL=fixedk)", status, "", &class));
        }
        let m = Module { handle };
        #[allow(unused_unsafe)]
        let f = unsafe { m.func(cuda, "im_bl_fixedk") }?;
        std::mem::forget(m);
        Ok(f as usize)
    }).clone().map(|u| u as *mut c_void)
}

impl Native {
    /// mode 0 tangent, 1 adjoint, in place on the shadows (the P3 convention). See the head of this file.
    unsafe fn im_p3_fixedk(&mut self, cuda: &Cuda, p: &mut Prepared, mode: i32) -> Result<Value, String> {
        if p.stripe.is_some() { return Err("AD_IM_BL=fixedk: a banded MYNN-EDMF bind (AD_P3_STRIPE) is not supported".into()); }
        let row = self.rows.get("rrfs_mynnedmf_run_l16").ok_or("AD_IM_BL=fixedk: the scheme bundle has no MYNN-EDMF row")?.clone();
        let g = |k: &str| -> Result<i64, String> { p.env.get(k).map(|v| *v as i64).ok_or_else(|| format!("AD_IM_BL=fixedk: MYNN-EDMF scalar {k}")) };
        let (first, count) = (g("batch_first")?, g("batch_count")?);
        let (its, ite, ims, ime, kms, kts, kte, jts, jte) =
            (g("its")?, g("ite")?, g("ims")?, g("ime")?, g("kms")?, g("kts")?, g("kte")?, g("jts")?, g("jte")?);
        if jts != jte { return Err(format!("AD_IM_BL=fixedk: the launch spans {} rows of j; one is supported", jte - jts + 1)); }
        let nk = kte - kts + 1;
        if !(2..=IM_NK_MAX).contains(&nk) { return Err(format!("AD_IM_BL=fixedk: {nk} levels; 2 to {IM_NK_MAX} are supported")); }
        let delt = *p.env.get("delt").ok_or("AD_IM_BL=fixedk: MYNN-EDMF scalar delt")? as f32;
        // kmode 1: MYNN's held K; 2: the boundary layer held (no mixing, no surface exchange)
        let bl_held = env_var("AD_IM_BL").as_deref() == Some("held");
        let kmode: i32 = if bl_held { 2 } else { 1 };
        let (heat, moist, mom) = match env_var("AD_IM_SFC").as_deref() {
            _ if bl_held => (0, 0, 0),
            None | Some("all") => (1, 1, 1),
            Some("momentum") => (0, 0, 1),
            Some("none") => (0, 0, 0),
            Some(o) => return Err(format!("AD_IM_SFC={o}: want all, momentum or none")),
        };
        // second form: AD_IM_COND = the condensation time scale in seconds (unset or 0: off, the first form)
        let cond_tau: f32 = match env_var("AD_IM_COND") {
            None => 0.0,
            Some(v) => v.parse::<f32>().ok().filter(|t| t.is_finite() && *t >= 0.0).ok_or_else(|| format!("AD_IM_COND={v}: want seconds (0 or more)"))?,
        };
        let pre = |n: &str| -> Result<u64, String> { let (al, off) = Self::dev_arg(&row, &*p, n)?; p.segs.pre(al, off) };
        let sh = |n: &str| -> Result<u64, String> { let (al, off) = Self::dev_arg(&row, &*p, n)?; p.segs.shadow(al, off) };
        // the held outputs: every float array the launch writes but the four tendencies
        let names: Vec<String> = row.raw["float_arrays"].as_array().ok_or("float_arrays")?.iter()
            .map(|v| v.as_str().unwrap_or("").to_owned()).collect();
        let (mut ptrs, mut dims) = (Vec::new(), Vec::new());
        let mut held = Vec::new();
        for n in names.iter().filter(|n| !IM_BL_INPUTS.contains(&n.as_str()) && !IM_BL_TEND.contains(&n.as_str())) {
            let a = row.arg(n)?;
            let Some((al, off)) = p.args.get(a.pos).and_then(target) else { continue };
            let Ok(s) = p.segs.shadow(al, off) else { continue };
            let three = a.extent.as_deref().is_some_and(|e| e.contains("kme"));
            ptrs.push(s);
            dims.push(if three { 3i32 } else { 2 });
            held.push(n.clone());
        }
        if ptrs.len() > IM_MAX_OUT { return Err(format!("AD_IM_BL=fixedk: {} held outputs, at most {IM_MAX_OUT}", ptrs.len())); }
        let mut outs: Vec<u8> = Vec::with_capacity(IM_MAX_OUT * 12 + 8);
        for k in 0..IM_MAX_OUT { outs.extend(ptrs.get(k).copied().unwrap_or(0).to_le_bytes()); }
        for k in 0..IM_MAX_OUT { outs.extend(dims.get(k).copied().unwrap_or(0).to_le_bytes()); }
        outs.extend((ptrs.len() as i32).to_le_bytes());
        while outs.len() % 8 != 0 { outs.push(0); }
        let args = vec![
            i32b(mode), i32b(first as i32), i32b(count as i32), i32b(its as i32), i32b(ite as i32), i32b(ims as i32),
            i32b(ime as i32), i32b(kms as i32), i32b(kts as i32), i32b(kte as i32), K::B(delt.to_le_bytes().to_vec()),
            i32b(heat), i32b(moist), i32b(mom), K::B(cond_tau.to_le_bytes().to_vec()),
            K::P(pre("tt")?), K::P(pre("p")?), K::P(pre("exner")?),
            K::P(pre("dz")?), K::P(pre("rho")?), i32b(kmode), K::P(pre("el_pbl")?), K::P(pre("qke")?),
            K::P(pre("sh3d")?), K::P(pre("sm3d")?), K::P(pre("ust")?),
            K::P(pre("wspd")?), K::P(pre("ch")?), K::P(pre("qfx")?), K::P(pre("qsfc")?), K::P(pre("qv")?),
            K::P(sh("th")?), K::P(sh("qv")?), K::P(sh("u")?), K::P(sh("v")?),
            K::P(sh("rthblten")?), K::P(sh("rqvblten")?), K::P(sh("rublten")?), K::P(sh("rvblten")?),
            K::B(outs),
        ];
        // AD_IM_DIAG=1 (diagnostic runs only): the held coefficients this launch reads, summarised on the host (one
        // read-back per launch, so it is off for every timed run).
        if env_var("AD_IM_DIAG").as_deref() == Some("1") {
            let ni = (ime - ims + 1) as usize;
            let nkm = (g("kme")? - kms + 1) as usize;
            let rd = |n: &str, words: usize| -> Result<Vec<f32>, String> {
                let b = dev::download(cuda, pre(n)?, words * 4)?;
                Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
            };
            let (eh, em, ust, chv) = (rd("exch_h", ni * nkm)?, rd("exch_m", ni * nkm)?, rd("ust", ni)?, rd("ch", ni)?);
            let (el, qk, shv, smv) = (rd("el_pbl", ni * nkm)?, rd("qke", ni * nkm)?, rd("sh3d", ni * nkm)?, rd("sm3d", ni * nkm)?);
            let kh: Vec<f32> = (0..el.len()).map(|j| el[j].max(0.0) * qk[j].max(0.0).sqrt() * shv[j].max(0.0)).collect();
            let stat = |v: &[f32]| { let f: Vec<f64> = v.iter().map(|x| *x as f64).filter(|x| x.is_finite()).collect();
                json!({"max": f.iter().cloned().fold(0.0, f64::max), "mean": f.iter().sum::<f64>() / f.len().max(1) as f64,
                       "positive": f.iter().filter(|x| **x > 0.0).count(), "n": f.len()}) };
            eprintln!("{}", json!({"event":"inner_model_bl_diag","mode":mode,"first":first,"count":count,
                "kmode":kmode,"exch_h":stat(&eh),"exch_m":stat(&em),"ust":stat(&ust),"ch":stat(&chv),
                "el_pbl":stat(&el),"qke":stat(&qk),"sh3d":stat(&shv),"sm3d":stat(&smv),"k_heat":stat(&kh)}));
        }
        let f = im_function(cuda)?;
        launch(cuda, f, grid1(count.max(0) as usize, 64), [64, 1, 1], &args)?;
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| eprintln!("{}", json!({"event":"inner_model_bl","route":if bl_held { "held" } else { "fixedk" },"k":"el sqrt(qke) sh|sm, last call's","surface":[heat, moist, mom],"condensation_tau_s":cond_tau,
            "held_outputs":held, "note":"said once; every boundary-layer launch of both sweeps runs the held-coefficient diffusion"})));
        Ok(json!({"route": "inner-model fixedk", "columns": count}))
    }
}
