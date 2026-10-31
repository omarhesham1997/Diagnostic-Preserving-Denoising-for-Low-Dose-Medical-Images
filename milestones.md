
### 6.1 Milestone 1 — Proposal & Literature Review Template

```latex
\documentclass[11pt]{article}

% ============================================================
% Milestone 1 -- Project Proposal & Literature Review Template
% DIP Team Project
% ============================================================
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[round]{natbib}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue!50!black, citecolor=blue!50!black, urlcolor=blue!50!black}
\usepackage{enumitem}
\usepackage{xcolor}

\newcommand{\todo}[1]{\textcolor{red}{\textbf{[TODO: #1]}}}

\title{\LARGE Project Proposal \& Literature Review \\[4pt]
\large \todo{Project Title}}
\author{
  \todo{Student A Full Name} \and
  \todo{Student B Full Name} \and
  \todo{Student C Full Name}
}
\date{Milestone 1 --- \todo{Date} \\{} [COURSE CODE] --- Digital Image Processing}

\begin{document}
\maketitle

\begin{abstract}
\todo{3--5 sentences: what problem are you tackling, why does it matter, what is your
planned approach, and what public dataset/baseline will you use. Write this last.}
\end{abstract}

\section{Problem Statement \& Motivation}
\label{sec:motivation}
\todo{What image-processing problem are you solving? Why is it interesting or useful?
How does it connect to material from the course (name the week/topic)? 1 paragraph.}

\section{Related Work}
\label{sec:related}
\todo{Summarize at least 4 references, at least 2 of which should be from 2023--2026.
Use the table below as a starting organizer, then write 1--2 paragraphs of prose
connecting them to your plan.}

\begin{table}[h]
\centering
\caption{Key references.}
\label{tab:refs}
\begin{tabular}{@{}p{0.30\linewidth}p{0.12\linewidth}p{0.45\linewidth}@{}}
\toprule
\textbf{Reference} & \textbf{Year} & \textbf{Relevance to this project} \\
\midrule
\citep{example2024} & 2024 & \todo{one sentence} \\
\citep{example2018}  & 2018 & \todo{one sentence} \\
\bottomrule
\end{tabular}
\end{table}

\section{Dataset \& Baseline Plan}
\label{sec:data}
\todo{Name the public dataset(s), give exact size/subset you will use, and the license.
State exactly which baseline (classical or prior learned method) you will reproduce
and compare against.}

\begin{table}[h]
\centering
\caption{Dataset summary.}
\label{tab:data}
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Dataset} & \textbf{Subset used} & \textbf{Link} \\
\midrule
\todo{name} & \todo{e.g., 800 train / 100 val crops} & \todo{URL} \\
\bottomrule
\end{tabular}
\end{table}

\section{Proposed Method}
\label{sec:method}
\todo{Sketch your planned approach: architecture diagram or pseudocode, what is novel
or extended relative to the baseline, and why you expect it to work.}

\section{Feasibility \& Compute Plan}
\label{sec:feasibility}
\todo{State your compute tier (CPU-OK / GPU-light), expected training time per
experiment, and confirm this fits a laptop-class machine within the project window.}

\subsection{Minimum Viable Deliverable (MVP)}
\todo{The smallest version of this project that still produces a valid, reportable
comparison against the baseline.}

\subsection{Stretch Goal}
\todo{What you will attempt if the MVP is on track by Week 5--6.}

\section{Risk Assessment}
\label{sec:risk}
\begin{itemize}[leftmargin=*]
  \item \todo{Risk 1 and its mitigation / fallback.}
  \item \todo{Risk 2 and its mitigation / fallback.}
\end{itemize}

\section{Work Breakdown}
\label{sec:workbreakdown}
\begin{table}[h]
\centering
\caption{Milestone task allocation (see also the project catalog entry for this
project).}
\label{tab:tasks}
\begin{tabular}{@{}p{0.12\linewidth}p{0.26\linewidth}p{0.26\linewidth}p{0.26\linewidth}@{}}
\toprule
\textbf{Milestone} & \textbf{Student A} & \textbf{Student B} & \textbf{Student C} \\
\midrule
M1 (Wk 3) & \todo{task} & \todo{task} & \todo{task} \\
M2 (Wk 6) & \todo{task} & \todo{task} & \todo{task} \\
M3 (Wk 9) & \todo{task} & \todo{task} & \todo{task} \\
\bottomrule
\end{tabular}
\end{table}

\bibliographystyle{unsrtnat}
\bibliography{references}

\end{document}
```

### 6.2 Milestone 2 — Progress Report Template

```latex
\documentclass[11pt]{article}

% ============================================================
% Milestone 2 -- Progress Report Template
% DIP Team Project
% ============================================================
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[round]{natbib}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue!50!black, citecolor=blue!50!black, urlcolor=blue!50!black}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{caption}

\newcommand{\todo}[1]{\textcolor{red}{\textbf{[TODO: #1]}}}

\title{\LARGE Progress Report \\[4pt]
\large \todo{Project Title}}
\author{
  \todo{Student A Full Name} \and
  \todo{Student B Full Name} \and
  \todo{Student C Full Name}
}
\date{Milestone 2 --- \todo{Date} \\{} [COURSE CODE] --- Digital Image Processing}

\begin{document}
\maketitle

\begin{abstract}
\todo{3--5 sentences: recap the problem, state what is implemented and working now,
and headline your single most important preliminary result.}
\end{abstract}

\section{Recap}
\label{sec:recap}
\todo{1 paragraph: problem, baseline, dataset -- carried over from M1, updated if scope
changed.}

\section{Implementation Status}
\label{sec:status}
\todo{What is built and working right now. Be concrete: architecture as actually
implemented (not as planned), training configuration, any deviation from the M1 plan
and why.}

\subsection{Baseline Reproduction}
\todo{Confirm the baseline is reproduced and working; report its numbers.}

\subsection{Your Method}
\todo{Status of your extension/contribution: what runs end-to-end, what doesn't yet.}

\section{Preliminary Results}
\label{sec:results}
\todo{At least one quantitative table or rate-distortion/accuracy curve comparing your
method (even partially trained) to the baseline, plus 2--3 qualitative examples.}

\begin{table}[h]
\centering
\caption{Preliminary quantitative comparison.}
\label{tab:prelim}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Method} & \textbf{Metric 1} & \textbf{Metric 2} & \textbf{Notes} \\
\midrule
Baseline        & \todo{val} & \todo{val} & \todo{} \\
Ours (partial)  & \todo{val} & \todo{val} & \todo{} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
\centering
% \includegraphics[width=0.8\linewidth]{figures/preliminary_result.png}
\caption{\todo{Describe what this figure actually shows -- not what you expected it to
show. If the qualitative result is weaker or different than hoped, say so and explain
your current hypothesis.}}
\label{fig:prelim}
\end{figure}

\section{Challenges \& Deviations from the M1 Plan}
\label{sec:challenges}
\todo{Be specific and honest. What did not work as planned? What did you change and
why? This section is graded on candor and problem-solving, not on everything having
gone smoothly.}

\section{Updated Work Breakdown \& MVP Status}
\label{sec:workbreakdown}
\todo{State explicitly whether the M1 minimum-viable deliverable has been reached. If
not, what is the revised plan to reach it before M3?}

\begin{table}[h]
\centering
\caption{Updated milestone task allocation.}
\label{tab:tasks}
\begin{tabular}{@{}p{0.12\linewidth}p{0.26\linewidth}p{0.26\linewidth}p{0.26\linewidth}@{}}
\toprule
\textbf{Milestone} & \textbf{Student A} & \textbf{Student B} & \textbf{Student C} \\
\midrule
M2 (Wk 6) -- completed & \todo{what was done} & \todo{what was done} & \todo{what was done} \\
M3 (Wk 9) -- planned   & \todo{task} & \todo{task} & \todo{task} \\
\bottomrule
\end{tabular}
\end{table}

\section{Next Steps}
\label{sec:next}
\todo{Concrete plan for Weeks 7--9.}

\bibliographystyle{unsrtnat}
\bibliography{references}

\end{document}
```

### 6.3 Milestone 3 — Final Report Template

```latex
\documentclass[11pt]{article}

% ============================================================
% Milestone 3 -- Final Report Template
% DIP Team Project
% ============================================================
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[round]{natbib}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue!50!black, citecolor=blue!50!black, urlcolor=blue!50!black}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{subcaption}

\newcommand{\todo}[1]{\textcolor{red}{\textbf{[TODO: #1]}}}

\title{\LARGE \todo{Project Title} \\[4pt]
\large Final Report}
\author{
  \todo{Student A Full Name} \and
  \todo{Student B Full Name} \and
  \todo{Student C Full Name}
}
\date{Milestone 3 --- \todo{Date} \\{} [COURSE CODE] --- Digital Image Processing}

\begin{document}
\maketitle

\begin{abstract}
\todo{150--250 words. Problem, approach, dataset, headline quantitative result vs.
the stated baseline/SOTA reference, and one-sentence takeaway. Write this last.}
\end{abstract}

\section{Introduction}
\label{sec:intro}
\todo{What problem, why it matters, what you did, and a one-paragraph summary of your
findings (reviewers should be able to read the intro and abstract alone and know what
you found).}

\section{Related Work}
\label{sec:related}
\todo{Position your work relative to the references identified at M1 (and any added
since). Group by theme, not by chronology. At least 2 references should be from
2023--2026.}

\section{Method}
\label{sec:method}
\todo{Describe your final approach precisely enough that another team could
reimplement it: architecture, loss function(s), training procedure, hyperparameters.
Use an algorithm block or architecture figure if it helps.}

\begin{figure}[h]
\centering
% \includegraphics[width=0.85\linewidth]{figures/architecture.png}
\caption{\todo{Architecture / pipeline diagram.}}
\label{fig:arch}
\end{figure}

\section{Experimental Setup}
\label{sec:setup}

\subsection{Dataset}
\todo{Exact subset used (count, selection method, random seed), license, and any
preprocessing.}

\subsection{Baseline(s)}
\todo{Exact baseline(s) you compare against, and confirmation that your reproduction
matches published numbers (or an explanation of any gap).}

\subsection{Metrics \& Protocol}
\todo{Every metric used, computed how, and the exact comparison protocol (e.g.,
bpp-matching procedure for rate--distortion comparisons, train/val/test split, number
of seeds).}

\section{Results}
\label{sec:results}

\begin{table}[h]
\centering
\caption{Main quantitative comparison against the baseline.}
\label{tab:main}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Method} & \textbf{Metric 1} & \textbf{Metric 2} & \textbf{Metric 3} & \textbf{Compute/Runtime} \\
\midrule
Baseline      & \todo{val} & \todo{val} & \todo{val} & \todo{val} \\
Ours          & \todo{val} & \todo{val} & \todo{val} & \todo{val} \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[h]
\centering
% \includegraphics[width=0.48\linewidth]{figures/qual1.png}
% \includegraphics[width=0.48\linewidth]{figures/qual2.png}
\caption{\todo{Qualitative comparison. Describe exactly what is shown, including any
failure cases -- do not caption a figure with what you expected it to show.}}
\label{fig:qual}
\end{figure}

\subsection{Ablation Studies}
\label{sec:ablation}
\todo{At least one ablation isolating the effect of your main design choice. Table or
plot required, not just prose.}

\section{Discussion \& Limitations}
\label{sec:discussion}
\todo{What worked, what didn't, and why (to the best of your understanding). Be
explicit about limitations -- scope, dataset size, compute budget, anything that would
change if this were scaled up. A paper that honestly analyzes its limitations is
graded higher than one that overclaims.}

\section{Conclusion}
\label{sec:conclusion}
\todo{2--4 sentences: what you built, what you found, what you would do next with more
time/compute.}

\section*{Individual Contributions}
\label{sec:contributions}
\todo{One sentence per student stating their primary contributions, consistent with
the peer-evaluation form submitted separately.}

\bibliographystyle{unsrtnat}
\bibliography{references}

\appendix
\section{Additional Results}
\label{sec:appendix}
\todo{Optional: extra qualitative examples, full hyperparameter tables, or results
that did not fit in the main body.}

\end{document}
```

---

## 7. Milestone Rubrics

These three rubrics apply **uniformly across all 30 catalog projects** (and any approved
custom topic) — no per-project rubric is needed, since "did you reproduce a credible
baseline, implement a scoped extension, and benchmark it fairly" is the same question
regardless of which image-processing problem a team picked.

**Weighted project grade** = 0.15 × M1 + 0.35 × M2 + 0.50 × M3 (see Section 4).

### 7.1 Milestone 1 Rubric — Proposal & Literature Review (100 pts)

| Criterion | Points | What earns full marks |
|---|---|---|
| Problem motivation & scope clarity | 15 | Clear statement of the problem, why it matters, and explicit connection to a course topic |
| Literature review | 25 | ≥4 references correctly summarized, ≥2 from 2023–2026, understanding demonstrated (not just citation-dropping) |
| Dataset & baseline identification | 15 | Public, correctly licensed, feasible dataset; baseline is a real, citable method |
| Feasibility & compute plan | 15 | Realistic for the stated compute tier and 9-week window; subsampling plan stated |
| Task breakdown & team role clarity | 10 | All 3 milestones' tasks assigned per student, consistent with Section 5 |
| Risk assessment & MVP/stretch split | 10 | At least 2 real risks with mitigations; MVP is genuinely minimal and still valid |
| Writing & LaTeX quality | 10 | Uses the Section 6.1 template correctly, proper `\citep`/`\citet`, clean tables/figures |

### 7.2 Milestone 2 Rubric — Implementation & Preliminary Results (100 pts)

| Criterion | Points | What earns full marks |
|---|---|---|
| Baseline reproduction correctness | 20 | Baseline runs end-to-end and its numbers are plausible/consistent with the literature |
| Core method implementation | 25 | The team's own method/extension runs end-to-end, even if not fully tuned |
| Preliminary quantitative results | 20 | At least one real comparison table/curve, correct metrics, computed correctly |
| Experimental rigor | 15 | Clean train/val/test split, fixed seed, `requirements.txt`/`environment.yml` present |
| Progress vs. M1 plan | 10 | MVP status stated explicitly; deviations explained, not hidden |
| Report quality | 10 | Uses the Section 6.2 template; figures/tables captioned accurately (not aspirationally) |

### 7.3 Milestone 3 Rubric — Final Report, Results & Presentation (100 pts)

| Criterion | Points | What earns full marks |
|---|---|---|
| Technical correctness & completeness | 20 | Method fully and precisely described; reproducible from the report alone |
| Quantitative comparison vs. baseline/SOTA | 20 | Fair, documented protocol; results support the claims made |
| Ablation studies / analysis depth | 15 | ≥1 ablation isolating the main design choice's effect |
| Reproducibility | 10 | Public code repo, README, fixed seeds, environment file, documented hyperparameters |
| Final report writing quality | 15 | Uses the Section 6.3 template; honest limitations section; correct citations |
| Presentation / demo | 10 | Clear 8–10 min presentation + live Q&A; working demo if applicable |
| Individual contribution clarity | 10 | Consistent with the peer-evaluation form (Section 9.5); every student can answer questions on the whole pipeline |

---

## 8. Publication Bonus Track

Some teams will produce work that is genuinely competitive beyond the course. This track
rewards that **without** lowering the bar for the standard rubrics above — bonus points are
additive, never a substitute for meeting the core milestone criteria.

### 8.1 Bonus Scorecard

| # | Criterion | Met? |
|---|---|---|
| 1 | **Novelty beyond reproduction** — a new method, combination, or analysis, not just a re-run of existing code | ☐ |
| 2 | **Matches or improves a stated recent SOTA baseline** on a public benchmark under a fair, documented protocol — *or* delivers a rigorous, well-analyzed negative/surprising result | ☐ |
| 3 | **At least one ablation study** isolating the contribution's effect | ☐ |
| 4 | **Full reproducibility** — public code repo, README, fixed seeds, environment file, documented hyperparameters | ☐ |
| 5 | **Venue-standard writing** — honest related-work positioning, explicit limitations, correct citations throughout | ☐ |
| 6 | **Clean academic-integrity record** — proper attribution of pretrained/frozen components, AI-tool usage disclosed per Section 9.1 | ☐ |

**Grading mechanics:**
- **≥5 of 6 criteria met:** up to **+5 percentage points** added to the project's final
  weighted score (capped at 100).
- **All 6 criteria met, and the instructor judges the work genuinely submission-ready:**
  the team is offered co-mentored submission to one of the venues below, with the instructor
  as corresponding/senior author per standard supervisory-authorship norms. This is offered,
  not automatic — **pre-approval from the instructor/TA is required before the M3 deadline**,
  so there are no last-minute surprises about whether a claim of novelty holds up.
