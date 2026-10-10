import io, re, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

st.set_page_config(page_title="Hispanic Heritage Data Lab", page_icon="🌎", layout="wide")
COUNTRIES={"Mexico":"MEX","Brazil":"BRA","Argentina":"ARG","Colombia":"COL","Chile":"CHL","Peru":"PER","Venezuela":"VEN","Ecuador":"ECU","Uruguay":"URY"}
INDICATORS={
"Life expectancy at birth":("SP.DYN.LE00.IN","years",0,125,"Expected years a newborn would live if current age-specific mortality rates persisted."),
"Unemployment rate (national estimate)":("SL.UEM.TOTL.NE.ZS","% of total labor force",0,100,"Unemployed people as a percentage of the labor force, using national estimates."),
"GDP per capita":("NY.GDP.PCAP.CD","current US$ per person",0,None,"GDP divided by midyear population; not average wealth or household income."),
"Crude birth rate":("SP.DYN.CBRT.IN","births per 1,000 people",0,100,"Annual live births per 1,000 people; a rate, not a count."),
"Net migration":("SM.POP.NETM","people",None,None,"Immigrants minus emigrants for the source period; not a count of people leaving."),
"Intentional homicide rate":("VC.IHR.PSRC.P5","homicides per 100,000 people",0,None,"Intentional homicide victims per 100,000 people."),
"Consumer price inflation":("FP.CPI.TOTL.ZG","% annual change",-100,None,"Annual percentage change in the consumer price index."),
"Central government debt":("GC.DOD.TOTL.GD.ZS","% of GDP",None,None,"Central government debt as a percentage of GDP, not all public-sector debt."),
"Total reserves, including gold":("FI.RES.TOTL.CD","current US dollars",0,None,"Total reserves including gold, in current US dollars."),
"External balance: goods and services":("NE.RSB.GNFS.ZS","% of GDP",None,None,"External balance on goods and services, not the current account.")
}
WB="https://api.worldbank.org/v2"
CONTEXTS={
"Business":("time","years","profit","dollars","x**2 - 4*x + 8","Illustrative profit-like quantity."),
"Engineering":("time","seconds","position","meters","0.5*x**2 + 2*x","For position, the first derivative is velocity and the second is acceleration."),
"Medical":("time","hours","concentration","mg/L","10*exp(-0.25*x)","Illustrative concentration only; not medical advice."),
"Sports science":("time","seconds","distance","meters","2*x - 0.08*x**2","Illustrative distance model; interpret only within its chosen interval."),
"Biology":("time","days","population","organisms","100*exp(0.12*x)","Illustrative population model."),
"Bioengineering":("time","seconds","signal","units","sin(x)","Illustrative periodic signal."),
"Economic forecasting":("year","years","indicator","index units","0.03*x**2 + 2*x + 10","Illustrative trend, not an official forecast."),
"Weather":("time","hours","temperature","°C","10 + 5*sin(2*pi*x/24)","Illustrative temperature cycle."),
"Custom":("x","user units","f(x)","user units","x**2","Enter a function in x.")
}
x=sp.symbols("x", real=True)
SAFE_FUNCS={"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,"log":sp.log,"sqrt":sp.sqrt,"abs":sp.Abs,"pi":sp.pi,"E":sp.E}
def parse_expr(text):
    if len(text)>250: raise ValueError("Function is too long.")
    if not re.fullmatch(r"[0-9xX+\-*/().,^ \tA-Za-z_]+",text): raise ValueError("Use mathematical notation with x, numbers, operators, and supported functions.")
    text=text.replace("^","**").replace("X","x")
    try: e=sp.sympify(text, locals={**SAFE_FUNCS,"x":x}, evaluate=True)
    except Exception as ex: raise ValueError("Could not parse that function. Try notation such as x**2, exp(x), log(x), sin(x).") from ex
    if not e.free_symbols.issubset({x}): raise ValueError("Only x is allowed as a variable.")
    if e.has(sp.zoo,sp.nan,sp.oo,-sp.oo): raise ValueError("Function contains an undefined or infinite expression.")
    return e
def eval_expr(expr, vals):
    fn=sp.lambdify(x,expr,modules=["numpy"])
    with np.errstate(all="ignore"):
        out=np.asarray(fn(vals),dtype=float)
    if out.shape==(): out=np.full(np.asarray(vals).shape,float(out))
    return out
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code, indicator_code, first_year, last_year):
    url=f"{WB}/country/{country_code}/indicator/{indicator_code}"
    r=requests.get(url,params={"date":f"{first_year}:{last_year}","format":"json","per_page":1000},timeout=18)
    r.raise_for_status()
    payload=r.json()
    if not isinstance(payload,list) or len(payload)<2 or not isinstance(payload[1],list):
        raise ValueError("The World Bank returned an unexpected response.")
    rows=[]
    for item in payload[1]:
        if item.get("value") is not None:
            rows.append((int(item["date"]),float(item["value"])))
    return pd.DataFrame(rows,columns=["Year","Value"]).sort_values("Year").reset_index(drop=True)
def fit_poly(df, degree):
    clean=df[["Year","Value"]].dropna().copy()
    clean["Year"]=pd.to_numeric(clean.Year,errors="coerce")
    clean["Value"]=pd.to_numeric(clean.Value,errors="coerce")
    clean=clean.dropna()
    if (clean.Year%1!=0).any(): raise ValueError("Years must be integers.")
    clean["Year"]=clean.Year.astype(int)
    if clean.Year.duplicated().any(): raise ValueError("Duplicate years are not allowed.")
    clean=clean.sort_values("Year")
    if len(clean)<degree+1: raise ValueError(f"Degree {degree} requires at least {degree+1} distinct observations.")
    base=int(clean.Year.min())
    coeff=np.polyfit(clean.Year.to_numpy()-base,clean.Value.to_numpy(),degree)
    pred=np.polyval(coeff,clean.Year.to_numpy()-base)
    mae=float(np.mean(np.abs(clean.Value.to_numpy()-pred)))
    ss_res=float(np.sum((clean.Value.to_numpy()-pred)**2)); ss_tot=float(np.sum((clean.Value.to_numpy()-clean.Value.mean())**2))
    r2=float(1-ss_res/ss_tot) if ss_tot>0 else float("nan")
    return clean,base,coeff,pred,mae,r2
def poly_text(coeff,base):
    terms=[]
    d=len(coeff)-1
    for i,c in enumerate(coeff):
        power=d-i
        terms.append(f"({c:.6g})·(year − {base})^{power}" if power>1 else (f"({c:.6g})·(year − {base})" if power==1 else f"({c:.6g})"))
    return " + ".join(terms)
def csv_bytes(df): return df.to_csv(index=False).encode("utf-8")
def excel_bytes(summary, data, results):
    out=io.BytesIO()
    with pd.ExcelWriter(out,engine="xlsxwriter") as writer:
        pd.DataFrame(summary.items(),columns=["Field","Value"]).to_excel(writer,index=False,sheet_name="Summary")
        data.to_excel(writer,index=False,sheet_name="Working data")
        results.to_excel(writer,index=False,sheet_name="Calculated years")
    return out.getvalue()
def pdf_bytes(title, meta, metrics, chart_png, data, results):
    out=io.BytesIO(); doc=SimpleDocTemplate(out,pagesize=letter,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36)
    styles=getSampleStyleSheet(); story=[Paragraph(title,styles["Title"]),Spacer(1,8)]
    for k,v in meta.items(): story.append(Paragraph(f"<b>{k}:</b> {str(v)}",styles["BodyText"]))
    story.append(Spacer(1,8))
    for k,v in metrics.items(): story.append(Paragraph(f"<b>{k}:</b> {str(v)}",styles["BodyText"]))
    story += [Spacer(1,10),Paragraph("Historical data and fitted model",styles["Heading2"]),Image(io.BytesIO(chart_png),width=6.7*inch,height=3.4*inch)]
    story.append(Paragraph("Data preview",styles["Heading2"]))
    preview=data.head(14).round(5).values.tolist()
    t=Table([["Year","Value"]]+preview,repeatRows=1,colWidths=[1.2*inch,2.0*inch]); t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.3,colors.grey)])); story.append(t)
    if not results.empty:
        story += [Spacer(1,8),Paragraph("Calculated-year results",styles["Heading2"])]
        rt=Table([list(results.columns)]+results.astype(str).values.tolist(),repeatRows=1)
        rt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),.3,colors.grey)])); story.append(rt)
    story.append(Spacer(1,10)); story.append(Paragraph("Limitations: polynomial fit is descriptive, R² is in-sample, extrapolations may be implausible, and validated prediction intervals are not provided.",styles["BodyText"]))
    doc.build(story); return out.getvalue()

st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("Turn authentic country indicators into mathematical investigations. Data source: World Bank API.")
st.info("Brazil is included as a Latin American comparison; it is not a Spanish-speaking Hispanic country. Use the data as one lens on history and society—not as a ranking of cultural worth.")
with st.sidebar:
    st.header("Historical data")
    country=st.selectbox("Country",list(COUNTRIES),index=0)
    indicator=st.selectbox("Indicator",list(INDICATORS),index=1)
    degree=st.slider("Polynomial degree",3,8,3)
    sample_step=st.slider("Model sample increment (years)",1,10,1)
    horizon=st.slider("Forward display horizon (years)",0,50,5)
    comparisons=st.multiselect("Compare same indicator with", [c for c in COUNTRIES if c!=country], default=["Colombia"] if country=="Mexico" else [])
    printer=st.toggle("Printer-friendly view",False)
    st.divider()
    st.header("Function & calculus lab")
    level=st.selectbox("Mathematics pathway",["Honors Algebra 2","Precalculus","Calculus"],index=2)
    context=st.selectbox("Context",list(CONTEXTS))
    suggested=CONTEXTS[context]
    use_fit=st.checkbox("Use historical curve of best fit as function",False)
    function_choice=st.radio("Function source",["Suggested function","Edit parameters / enter function"],index=0)
    function_text=st.text_input("Function f(x)",value=suggested[4],help="Use x, + - * /, parentheses, ** or ^, sin, cos, tan, exp, log, sqrt, pi. Example: 2*x**2 - 3*x + 1")
    context_desc=st.text_area("Context description",value=suggested[5],height=75)
    indep=st.text_input("Independent variable",suggested[0])
    indep_unit=st.text_input("Independent units",suggested[1])
    dep=st.text_input("Dependent variable",suggested[2])
    dep_unit=st.text_input("Dependent units",suggested[3])
    lo=st.number_input("Analysis interval start",value=0.0,step=1.0)
    hi=st.number_input("Analysis interval end",value=10.0,step=1.0)
    show_f=st.checkbox("Show f(x)",True); show_d1=st.checkbox("Show first derivative",True); show_d2=st.checkbox("Show second derivative",level=="Calculus")
    point=st.slider("Current x value",float(min(lo,hi)),float(max(lo,hi)) if hi!=lo else float(lo+1),float(np.clip(2,min(lo,hi),max(lo,hi))) if hi!=lo else float(lo),step=0.1)
    left=st.number_input("Average-rate interval start",value=float(lo))
    right=st.number_input("Average-rate interval end",value=float(hi))
    auto=st.toggle("Automatic playback",False)
    speed=st.slider("Playback speed (seconds per step)",0.1,1.5,0.4,0.1)
    visual=st.selectbox("Contextual visual",["Walking / prosthetic leg","Moving object / projectile","Tank or concentration","Population / cell colony","Thermometer / weather","Business / economic indicator"])
    if st.button("Reset playback point"): st.session_state["play_x"]=float(lo)

first=max(1950,datetime.now().year-70); last=datetime.now().year-1
st.write(f"**Requested data window:** {first}–{last} (70 calendar years). Available observations may be fewer.")
code,units,minimum,maximum,definition=INDICATORS[indicator]
st.markdown(f"**Indicator definition:** {definition}  \n**Official code:** `{code}` · **Units:** {units}  \n**Source:** [World Bank indicator metadata](https://data.worldbank.org/indicator/{code}) · [API](https://api.worldbank.org/v2/country/{COUNTRIES[country]}/indicator/{code})")
load_col,reset_col=st.columns([4,1])
with load_col:
    with st.spinner(f"Retrieving {indicator} for {country} from the World Bank…"):
        try:
            source_df=fetch_world_bank(COUNTRIES[country],code,first,last)
            source_error=None
        except Exception as e:
            source_df=pd.DataFrame(columns=["Year","Value"]); source_error=str(e)
with reset_col:
    st.write("")
    reset_clicked=st.button("Reset working data")
if source_error:
    st.error(f"Live data retrieval failed: {source_error}. No replacement or fabricated values have been generated. Check the connection or retry later.")
if source_df.empty and not source_error:
    st.warning("The World Bank returned no non-null observations for this country, indicator, and window.")
if "working_key" not in st.session_state or st.session_state.working_key!=(country,indicator):
    st.session_state.working_key=(country,indicator)
    st.session_state.working_df=source_df.copy()
if reset_clicked:
    st.session_state.working_df=source_df.copy()
working=st.session_state.working_df.copy()
edited=st.data_editor(working,num_rows="dynamic",use_container_width=True,key=f"editor_{country}_{code}",column_config={"Year":st.column_config.NumberColumn("Year",step=1,format="%d"),"Value":st.column_config.NumberColumn(f"Value ({units})",format="%.6f")})
validation_messages=[]
valid=edited.copy()
valid["Year"]=pd.to_numeric(valid.get("Year"),errors="coerce")
valid["Value"]=pd.to_numeric(valid.get("Value"),errors="coerce")
bad=valid[valid.Year.isna()|valid.Value.isna()]
if not bad.empty: validation_messages.append(f"{len(bad)} row(s) with missing/non-numeric year or value will be discarded.")
valid=valid.dropna(subset=["Year","Value"])
valid=valid[valid.Year%1==0]
valid["Year"]=valid["Year"].astype(int)
if valid.Year.duplicated().any(): validation_messages.append("Duplicate years found. Fitting is blocked until each year is unique.")
if minimum is not None and (valid.Value<minimum).any(): validation_messages.append(f"Values below the indicator minimum ({minimum}) are invalid.")
if maximum is not None and (valid.Value>maximum).any(): validation_messages.append(f"Values above the indicator maximum ({maximum}) are invalid.")
if valid.Year.duplicated().any() or (minimum is not None and (valid.Value<minimum).any()) or (maximum is not None and (valid.Value>maximum).any()):
    model_df=pd.DataFrame(columns=["Year","Value"])
else: model_df=valid.sort_values("Year").reset_index(drop=True)
for m in validation_messages: st.warning(m)
edited_scenario=not (len(model_df)==len(source_df) and model_df.reset_index(drop=True).equals(source_df.reset_index(drop=True)))
if edited_scenario: st.warning("User-modified scenario: the working data differ from the retrieved source. Exports will be labeled accordingly.")
st.caption(f"Retrieved {len(source_df)} observations; working table has {len(model_df)} valid rows. Source retrieval timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.")
if not model_df.empty:
    st.dataframe(model_df,use_container_width=True)
    st.write(f"**Actual coverage:** {model_df.Year.min()}–{model_df.Year.max()} · **Observations:** {len(model_df)}")
else:
    st.info("A valid working series is needed for historical fitting.")

fit_ok=False
if len(model_df)>=degree+1 and not model_df.Year.duplicated().any():
    try:
        clean,base,coeff,pred,mae,r2=fit_poly(model_df,degree); fit_ok=True
    except Exception as e: st.error(f"Model fitting failed: {e}")
else:
    st.warning(f"Degree {degree} needs at least {degree+1} distinct valid years. Current valid rows: {len(model_df)}.")
if fit_ok:
    model_y=np.polyval(coeff,clean.Year.to_numpy()-base)
    residual=clean.Value.to_numpy()-model_y
    equation=poly_text(coeff,base)
    metrics={"Base year":base,"Degree":degree,"Observations":len(clean),"Mean":f"{clean.Value.mean():.6g} {units}","R-squared (in-sample)":f"{r2:.6g}","MAE (in-sample)":f"{mae:.6g} {units}","Equation":equation}
    st.subheader("Model summary")
    a,b,c,d=st.columns(4); a.metric("Observations",len(clean)); b.metric("R² · in-sample",f"{r2:.4f}"); c.metric("MAE · in-sample",f"{mae:.4g}"); d.metric("Fitted range",f"{clean.Year.min()}–{clean.Year.max()}")
    st.write(f"**Centered model:** let `t = year − {base}`.  \n`ŷ = {equation}`")
    st.caption("R² and MAE describe the in-sample fit; they do not establish forecasting accuracy. Residual error is not a validated prediction interval.")
    residual_df=clean.copy(); residual_df["Fitted"]=model_y; residual_df["Residual"]=residual
    with st.expander("Residual table and chronological holdout"):
        st.dataframe(residual_df,use_container_width=True)
        if len(clean)>=12:
            ntest=max(3,int(np.ceil(len(clean)*0.2)))
            train=clean.iloc[:-ntest]; test=clean.iloc[-ntest:]
            if len(train)>=degree+1:
                ctrain=np.polyfit(train.Year.to_numpy()-base,train.Value.to_numpy(),degree)
                hpred=np.polyval(ctrain,test.Year.to_numpy()-base)
                st.write(f"Chronological holdout: last {len(test)} observations, model fit only to earlier years. Holdout MAE = {np.mean(np.abs(test.Value.to_numpy()-hpred)):.6g} {units}.")
            else: st.write("Insufficient earlier observations to fit this degree and reserve a holdout.")
        else: st.write("Insufficient observations for a three-point chronological holdout at this degree; forecast skill is not tested.")
    # same-indicator country comparison
    comp_data={}
    for cname in comparisons:
        try: comp_data[cname]=fetch_world_bank(COUNTRIES[cname],code,first,last)
        except Exception as e: st.warning(f"{cname} comparison unavailable: {e}")
    # graph
    fig,ax=plt.subplots(figsize=(10,4.8))
    ax.scatter(clean.Year,clean.Value,label=f"{country} observed",s=25)
    smooth_end=int(clean.Year.max()+horizon)
    smooth_x=np.linspace(clean.Year.min(),max(clean.Year.max(),smooth_end),max(200,(smooth_end-int(clean.Year.min())+1)*2))
    smooth_y=np.polyval(coeff,smooth_x-base)
    inside=smooth_x<=clean.Year.max(); ax.plot(smooth_x[inside],smooth_y[inside],label="Fitted curve")
    if (~inside).any(): ax.plot(smooth_x[~inside],smooth_y[~inside],"--",label="Extrapolated segment")
    ax.axvline(clean.Year.max(),color="grey",linestyle=":",label="Historical-range boundary")
    sm=clean.iloc[::sample_step]
    ax.scatter(sm.Year,np.polyval(coeff,sm.Year-base),marker="x",label=f"Sampled model markers (every {sample_step} year(s))")
    for cname,dfc in comp_data.items():
        if not dfc.empty: ax.plot(dfc.Year,dfc.Value,marker=".",linestyle="-",label=f"{cname} observed ({dfc.Year.min()}–{dfc.Year.max()})")
        else: st.info(f"No available {indicator} observations for {cname} in the requested window.")
    ax.set_title(f"{indicator} — {country}"); ax.set_xlabel("Year"); ax.set_ylabel(units); ax.grid(True,alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    st.subheader("Historical observations, fit, extrapolation & comparison")
    if printer: st.markdown("### Printable research summary"); st.write(f"{country} · {indicator} · {units} · Source: World Bank · edited scenario: {edited_scenario}")
    st.pyplot(fig,use_container_width=True)
    png=io.BytesIO(); fig.savefig(png,format="png",dpi=160,bbox_inches="tight"); chart_png=png.getvalue(); plt.close(fig)
    # year estimates
    st.subheader("Estimate a year")
    col1,col2=st.columns([1,3])
    with col1:
        single_year=st.number_input("Year (1950–2100)",1950,2100, min(2100,max(1950,int(clean.Year.max()))),1,key="single_year")
        calculate=st.button("Calculate value")
    if calculate:
        val=float(np.polyval(coeff,int(single_year)-base))
        classification="within fitted historical range" if clean.Year.min()<=single_year<=clean.Year.max() else "outside fitted historical range (extrapolation)"
        st.session_state.single_result={"Year":int(single_year),"Estimate":val,"Units":units,"Classification":classification}
    single_result=st.session_state.get("single_result")
    if single_result and "Year" in single_result:
        st.write(f"**{single_result['Year']}:** {single_result['Estimate']:.8g} {units} — {single_result['Classification']}.")
        obs=clean.loc[clean.Year==single_result["Year"],"Value"]
        if len(obs): st.write(f"Observed source/working value at this year: {obs.iloc[0]:.8g} {units}; model estimate is a fitted value, not a replacement for the observation.")
    st.subheader("Batch estimates")
    mode=st.radio("Batch input",["Year range","Custom years"],horizontal=True)
    batch_years=[]; invalid_entries=[]
    if mode=="Year range":
        bc1,bc2,bc3=st.columns(3)
        with bc1: start=int(st.number_input("Start year",1950,2100,2000,1))
        with bc2: end=int(st.number_input("End year",1950,2100,2025,1))
        with bc3: step=int(st.number_input("Step",1,10,5,1))
        if end<=start: st.warning("End year must be greater than start year.")
        else: batch_years=list(range(start,end+1,step))
        if batch_years and batch_years[-1]!=end: st.caption("End year is included only if the step lands on it.")
    else:
        custom=st.text_input("Comma-separated years", "2000, 2010, 2020, 2030")
        for token in custom.split(","):
            token=token.strip()
            if not token: continue
            if not re.fullmatch(r"\d{4}",token): invalid_entries.append(token)
            elif not 1950<=int(token)<=2100: invalid_entries.append(token)
            else: batch_years.append(int(token))
        batch_years=list(dict.fromkeys(batch_years))
        if invalid_entries: st.warning("Ignored invalid or out-of-range entries: "+", ".join(invalid_entries))
    st.write("**Valid years:** "+(", ".join(map(str,batch_years)) if batch_years else "None"))
    batch_click=st.button("Calculate batch",disabled=not batch_years)
    if batch_click:
        rows=[]
        for yy in batch_years:
            rows.append({"Year":yy,"Estimate":float(np.polyval(coeff,yy-base)),"Units":units,"Classification":"within fitted range" if clean.Year.min()<=yy<=clean.Year.max() else "extrapolation"})
        st.session_state.batch_result=pd.DataFrame(rows)
        st.session_state.batch_key=(country,indicator,degree,tuple(map(tuple,model_df[["Year","Value"]].to_numpy())))
    results=st.session_state.get("batch_result",pd.DataFrame())
    current_key=(country,indicator,degree,tuple(map(tuple,model_df[["Year","Value"]].to_numpy())))
    if not isinstance(results,pd.DataFrame): results=pd.DataFrame()
    if st.session_state.get("batch_key")!=current_key: results=pd.DataFrame()
    st.dataframe(results,use_container_width=True) if not results.empty else None
    # rate of change and calculus/function analysis
    st.header("Function lab: rates, derivatives and contextual interpretation")
    try:
        if use_fit:
            expr=sum(sp.Float(float(cval),15)*(x-base)**(len(coeff)-i-1) for i,cval in enumerate(coeff))
            f_label=f"World Bank best-fit polynomial centered at {base}"
        else:
            expr=parse_expr(function_text)
            f_label=function_text
        d1=sp.diff(expr,x); d2=sp.diff(d1,x)
        st.latex("f(x)="+sp.latex(expr))
        if level=="Calculus":
            st.latex("f'(x)="+sp.latex(d1)); st.latex("f''(x)="+sp.latex(d2))
        elif level=="Precalculus":
            st.latex("f'(x)="+sp.latex(d1))
            st.caption("Precalculus view emphasizes how coefficients, rates, turning behavior, and concavity shape the model.")
        else: st.caption("Algebra 2 view emphasizes polynomial shape, domain/range, end behavior, and average rates.")
        if hi<=lo: st.warning("Analysis interval end must be greater than its start.")
        else:
            fx=float(eval_expr(expr,np.array([point]))[0])
            fleft=float(eval_expr(expr,np.array([left]))[0]); fright=float(eval_expr(expr,np.array([right]))[0])
            avg=(fright-fleft)/(right-left) if right>left else float("nan")
            slope=float(eval_expr(d1,np.array([point]))[0])
            curvature=float(eval_expr(d2,np.array([point]))[0])
            st.write(f"**{dep} at x = {point:.3g}:** {fx:.6g} {dep_unit}")
            st.write(f"**Instantaneous rate f′(x):** {slope:.6g} {dep_unit}/{indep_unit or 'x-unit'}. **Second derivative:** {curvature:.6g} {dep_unit}/{indep_unit or 'x-unit'}².")
            if right>left: st.write(f"**Average rate of change** from {left:g} to {right:g}: ({fright:.6g} − {fleft:.6g}) / ({right:g} − {left:g}) = **{avg:.6g} {dep_unit}/{indep_unit}**.")
            st.write(f"**Context:** {context_desc} At this point, the modeled {dep.lower()} is {fx:.5g} {dep_unit}; the local rate is {slope:.5g} {dep_unit} per {indep_unit}; the rate itself changes at {curvature:.5g} {dep_unit} per {indep_unit}².")
            # analysis graph with discontinuity gaps
            gx=np.linspace(float(lo),float(hi),700)
            gy=eval_expr(expr,gx); gy[~np.isfinite(gy)]=np.nan
            g1=eval_expr(d1,gx); g1[~np.isfinite(g1)]=np.nan
            g2=eval_expr(d2,gx); g2[~np.isfinite(g2)]=np.nan
            fig2,ax2=plt.subplots(figsize=(10,4.5))
            if show_f: ax2.plot(gx,gy,label="f(x)")
            if show_d1: ax2.plot(gx,g1,label="f′(x)")
            if show_d2: ax2.plot(gx,g2,label="f″(x)")
            ypoint=float(eval_expr(expr,np.array([point]))[0])
            if np.isfinite(ypoint): ax2.scatter([point],[ypoint],s=55,label="Current point")
            if right>left:
                yy1=float(eval_expr(expr,np.array([left]))[0]); yy2=float(eval_expr(expr,np.array([right]))[0])
                if np.isfinite(yy1) and np.isfinite(yy2):
                    ax2.plot([left,right],[yy1,yy2],"--",label="Secant line")
            dp=float(eval_expr(d1,np.array([point]))[0])
            if np.isfinite(ypoint) and np.isfinite(dp):
                tx=np.linspace(max(float(lo),point-(hi-lo)*.12),min(float(hi),point+(hi-lo)*.12),50)
                ax2.plot(tx,ypoint+dp*(tx-point),":",label="Tangent line")
            ax2.set_xlabel(f"{indep} ({indep_unit})"); ax2.set_ylabel(f"{dep} ({dep_unit})"); ax2.set_title(f"{context}: function and derivatives"); ax2.grid(True,alpha=.25); ax2.legend(); fig2.tight_layout()
            st.pyplot(fig2,use_container_width=True); plt.close(fig2)
            if level=="Calculus":
                st.subheader("Numerical calculus investigation")
                dense=np.linspace(float(lo),float(hi),2500)
                vals=eval_expr(expr,dense); vals[~np.isfinite(vals)]=np.nan
                der=eval_expr(d1,dense); der[~np.isfinite(der)]=np.nan
                der2=eval_expr(d2,dense); der2[~np.isfinite(der2)]=np.nan
                critical=[]
                for j in range(1,len(dense)):
                    if np.isfinite(der[j-1]) and np.isfinite(der[j]) and der[j-1]*der[j]<0:
                        critical.append(float((dense[j-1]+dense[j])/2))
                if critical:
                    rows=[]
                    for q in critical:
                        eps=max((hi-lo)/10000,1e-5)
                        l=float(eval_expr(d1,np.array([q-eps]))[0]); r=float(eval_expr(d1,np.array([q+eps]))[0])
                        kind="local minimum (derivative changes − to +)" if l<0<r else ("local maximum (derivative changes + to −)" if l>0>r else "stationary point; no sign-change classification")
                        rows.append({"x (approx.)":q,"f(x)":float(eval_expr(expr,np.array([q]))[0]),"classification":kind})
                    st.write("Critical points found by numerical sign-change search (approximations; stationary points without sign change may be missed):")
                    st.dataframe(pd.DataFrame(rows),use_container_width=True)
                else: st.write("No derivative sign-change critical points detected on this interval; this numerical scan can miss repeated roots.")
                inflect=[]
                for j in range(1,len(dense)):
                    if np.isfinite(der2[j-1]) and np.isfinite(der2[j]) and der2[j-1]*der2[j]<0:
                        inflect.append(float((dense[j-1]+dense[j])/2))
                if inflect:
                    st.write("Candidate inflection points (approximate; second-derivative sign changes verified numerically):")
                    st.dataframe(pd.DataFrame([{"x (approx.)":q,"f(x)":float(eval_expr(expr,np.array([q]))[0])} for q in inflect]),use_container_width=True)
                else: st.write("No concavity-change inflection points detected on the selected interval.")
                endpoints=[float(lo),float(hi)]
                candidates=endpoints+critical
                candidates=[q for q in candidates if np.isfinite(float(eval_expr(expr,np.array([q]))[0]))]
                if candidates:
                    vv=[float(eval_expr(expr,np.array([q]))[0]) for q in candidates]
                    st.write(f"On the sampled closed interval, candidate absolute minimum ≈ {min(vv):.6g} at x≈{candidates[int(np.argmin(vv))]:.6g}; candidate absolute maximum ≈ {max(vv):.6g} at x≈{candidates[int(np.argmax(vv))]:.6g}. Numerical search; discontinuities or missed critical points can affect conclusions.")
                st.caption("Domain restrictions are determined by the expression; plotted undefined/infinite values are left as gaps. Numerical root searches are approximations and do not prove exact extrema or inflection points.")
            # visual simulation
            st.subheader("Contextual visual simulation")
            phase=float(point)
            if auto:
                if "play_x" not in st.session_state: st.session_state.play_x=float(point)
                st.session_state.play_x += max((hi-lo)/100,0.05)
                if st.session_state.play_x>hi: st.session_state.play_x=float(lo)
                phase=st.session_state.play_x
                st.info(f"Playback point x ≈ {phase:.2f}; enable/disable Automatic playback to pause. Use Reset playback point to reset.")
                time.sleep(speed); st.rerun()
            scaled=float(np.clip((phase-lo)/(hi-lo),0,1)) if hi>lo else 0.5
            if visual=="Walking / prosthetic leg":
                st.markdown(f"**Walking-cycle illustration** · phase {scaled:.2f}.")
                st.markdown(f"<div style='font-size:48px;text-align:center'>🚶 {'🦿' if scaled<0.5 else '🦿'} ─────────→</div>",unsafe_allow_html=True)
            elif visual=="Moving object / projectile":
                st.markdown(f"**Moving object** · horizontal position scaled to the selected interval ({scaled:.2f}).")
                st.markdown(f"<div style='font-size:42px;text-align:center'>⚽ {'　' * int(scaled*12)}●</div>",unsafe_allow_html=True)
            elif visual=="Tank or concentration": st.progress(scaled,text=f"Illustrative level / concentration position: {scaled:.0%} of display scale")
            elif visual=="Population / cell colony": st.markdown("**Population / cell colony**"); st.markdown("🦠 " * (1+int(scaled*24)))
            elif visual=="Thermometer / weather": st.markdown(f"**Thermometer / weather indicator** · normalized display position {scaled:.0%}"); st.progress(scaled)
            else: st.markdown(f"**Business / economic indicator** · normalized display position {scaled:.0%}"); st.progress(scaled)
            st.caption("Context visuals are schematic and normalized to the selected interval; they do not represent measured biomechanics, dosage, physical trajectories, or real-world scale.")
    except Exception as e:
        st.error(f"Function error: {e}")
        st.info("Try a valid function such as x**2, exp(-0.25*x), log(x), sin(x), or 0.5*x**2 + 2*x. Logarithms and rational functions may be undefined on part of the selected interval.")

    # downloads
    st.header("Download reports")
    summary={"Country":country,"Country code":COUNTRIES[country],"Indicator":indicator,"Indicator code":code,"Units":units,"Definition":definition,"Source":"World Bank API","Source URL":f"https://data.worldbank.org/indicator/{code}","Retrieved UTC":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),"Requested window":f"{first}-{last}","Actual coverage":f"{clean.Year.min()}-{clean.Year.max()}","Observation count":len(clean),"Edited scenario":edited_scenario,"Degree":degree,"Base year":base,"Equation":equation,"R-squared in-sample":r2,"MAE in-sample":mae,"Validated prediction intervals":"Not provided"}
    result_export=results.copy() if not results.empty else pd.DataFrame(columns=["Year","Estimate","Units","Classification"])
    c1,c2,c3=st.columns(3)
    with c1: st.download_button("Download CSV data",csv_bytes(model_df),file_name=f"{country.lower()}_{code.lower()}_working_data.csv",mime="text/csv")
    with c2: st.download_button("Download Excel workbook",excel_bytes(summary,model_df,result_export),file_name=f"{country.lower()}_{code.lower()}_research.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3: st.download_button("Download PDF report",pdf_bytes("Hispanic Heritage Historical Data Explorer",summary,metrics,chart_png,model_df,result_export),file_name=f"{country.lower()}_{code.lower()}_report.pdf",mime="application/pdf")
    st.subheader("Research interpretation checklist")
    st.markdown("- What historical or policy changes might help explain this pattern? Support any claim with an independent credible source.\n- Are the two countries' definitions and coverage comparable?\n- Compare at least two polynomial degrees; a larger degree can improve in-sample fit without improving forecasting.\n- Calculate two average rates of change and distinguish percentage points per year from relative percent change where relevant.\n- Discuss three supported findings, two limitations, and one follow-up question. Do not infer causation from a curve alone.")
else:
    st.info("Model charts, estimates, and downloads become available after enough valid observations are available for the selected polynomial degree.")
st.caption("Research note: use at least two Spanish-speaking countries for the showcase; cite two historical/cultural sources in addition to primary data. This app does not replace source verification or teacher-assigned written analysis.")
