\ProvidesClass{tibop-article}[2021-09-21]
\usepackage[hyphens]{url}
\LoadClass[]{scrartcl}
\RequirePackage[a4paper, top=0.98in, bottom=0.79in, left=0.98in, right=0.98in]{geometry}
\RequirePackage[english]{babel}
\RequirePackage[]{silence}
\WarningFilter{biblatex}{'babel/polyglossia'} %We are not interested in the warning about csquotes
\RequirePackage{graphicx}
\RequirePackage{tabularx}
\RequirePackage{enumitem}
\RequirePackage[labelsep=period]{caption}
\RequirePackage{xcolor}
\RequirePackage[breaklinks=true, colorlinks=true, linkcolor=black, anchorcolor=blue, citecolor=black, filecolor=blue, menucolor=blue,  pagecolor=blue, urlcolor=blue]{hyperref}
\RequirePackage[backend=biber, style=ieee]{biblatex} %For bibliography management with biblatex
\usepackage{orcidlink}


\RequirePackage{ifthen}
\RequirePackage[headsepline]{scrlayer-scrpage}

\RequirePackage{kvoptions}
\newbool{isArticle}
\boolfalse{isArticle}
\DeclareVoidOption{journalarticle}{\booltrue{isArticle}}
\ProcessKeyvalOptions*

% ==========[ FONT ]============================================================


\RequirePackage{ifxetex}
\ifxetex
    \RequirePackage{fontspec}
    \setmainfont{Arimo}
\else
\RequirePackage{helvet}
    \renewcommand{\familydefault}{\sfdefault}
    \RequirePackage[utf8]{inputenc}
\fi

%===========[ Constants ]==========================================================
\renewcommand{\and}{, }
\newcommand{\lastand}{, and }
\newcommand{\affOne}{\textsuperscript{1}}
\newcommand{\affTwo}{\textsuperscript{2}}
\newcommand{\affThree}{\textsuperscript{3}}
\newcommand{\affFour}{\textsuperscript{4}}
\newcommand{\affFive}{\textsuperscript{5}}


\addto\captionsenglish{\renewcommand{\figurename}{Figure}}
\addto\captionsenglish{\renewcommand{\tablename}{Table}}
\RequirePackage{floatrow}	% Centers all flaots.
\floatsetup[table]{capposition=top} % For adjusting the position of the caption (default is below).

\defbibheading{references}{\section*{References}\sloppy}

% \newcommand{\orcid}[1]{\href{https://orcid.org/#1}{\textsuperscript{[#1]}}}
\newcommand{\orcid}[1]{\href{https://orcid.org/#1}{\textsuperscript{[https://orcid.org/#1]}}}
\pagenumbering{gobble}
\setlength{\parskip}{6pt}
\setlist[1]{topsep=0pt,itemsep=-5pt}

%===========[ Variables ]==========================================================
\newcommand{\contribtitle}{set title with \textbackslash TIBtitle in preamble}
\newcommand{\contribsubtitle}{}
\newcommand{\affiliations}{set affiliation with {\ttfamily \textbackslash TIBaffiliations\{\textbackslash AffOne X\textbackslash\textbackslash\textbackslash AffTwo Y\}} in preamble}

\newcommand{\bundlename}{journal/conference name}
\newcommand{\sessionname}{\ifbool{isArticle}{}{TIBconferencesession}}
\newcommand{\articleDOI}{TIB-OP will set DOI with \textbackslash TIBdoi}
\newcommand{\submitteddate}{1970-01-01}
\newcommand{\accepteddate}{1970-01-01}
\newcommand{\publisheddate}{1970-01-01}


\newcommand{\authorheaderstring}{}
\newcommand{\conferenceheaderstring}{}

\newcommand{\keywords}{}
\newcommand{\thisabstract}{}

%===========[ Setters ]==========================================================
\newcommand{\TIBkeywords}[1]{\renewcommand{\keywords}{#1}}
\newcommand{\TIBabstract}[1]{\renewcommand{\thisabstract}{#1}}
\newcommand{\TIBbundlename}[2][]{\renewcommand{\bundlename}{#2}\ifthenelse{\equal{#1}{}}{\renewcommand{\conferenceheaderstring}{#2}}{\renewcommand{\conferenceheaderstring}{#1}}}
\newcommand{\TIBconferencesession}[1]{\renewcommand{\sessionname}{#1}}
\newcommand{\TIBdoi}[1]{\renewcommand{\articleDOI}{#1}}
\newcommand{\TIBpublisheddate}[1]{\renewcommand{\publisheddate}{#1}}
\newcommand{\TIBaccepteddate}[1]{\renewcommand{\accepteddate}{#1}}
\newcommand{\TIBsubmitteddate}[1]{\renewcommand{\submitteddate}{#1}}
\newcommand{\TIBdate}[3]{\TIBpublisheddate{#1}\TIBaccepteddate{#2}\TIBsubmitteddate{#3}}
\newcommand{\TIBauthor}[2][]{
    \author{#2}
    \ifthenelse{\equal{#1}{}}
        {
         \renewcommand{\authorheaderstring}{#2}
         }
         {\renewcommand{\authorheaderstring}{#1}}
}
\newcommand{\TIBaffiliations}[1]{\renewcommand{\affiliations}{#1}}
\newcommand{\TIBtitle}[1]{\renewcommand{\contribtitle}{#1}}
\newcommand{\TIBsubtitle}[1]{\renewcommand{\contribsubtitle}{#1}}

\chead{\scriptsize\authorheaderstring\quad\textbar\quad\conferenceheaderstring}

%===========[ FORMAT ]==========================================================

\RedeclareSectionCommand[
    beforeskip=-10pt,
    afterskip=5pt,
font={\normalfont\Large\bfseries}
]{section}
\RedeclareSectionCommand[%
    beforeskip=-10pt,
    afterskip=5pt,
font={\normalfont\large\bfseries}
]{subsection}
\RedeclareSectionCommand[%
    beforeskip=-10pt,
    afterskip=5pt,
font={\normalfont\normalsize\bfseries}
]{subsubsection}
\RedeclareSectionCommand[%
    beforeskip=-10pt,
    afterskip=5pt,
font={\normalfont\itshape}
]{paragraph}

% \renewcommand*{\sectionformat}{}%no numbers
%\renewcommand*{\subsectionformat}{}%no numbers
%\renewcommand*{\subsubsectionformat}{}%no numbers
%\renewcommand*{\paragraphformat}{}%no numbers

\widowpenalty=1000000
\clubpenalty=1000000

\captionsetup{belowskip=0pt,skip=6pt,labelfont={bf,small,it},textfont={small,it}}%justification=raggedright,singlelinecheck=false
\makeatletter
\renewcommand{\maketitle}{
        \footnotesize\noindent
        {\bundlename}\medskip\\
        \ifbool{isArticle}
            {}
            {{\sessionname}\medskip\\}
        %endbool isArticle
        {\articleDOI}\medskip\\
        {\textcopyright}\ Authors. This work is licensed under a \href{https://creativecommons.org/licenses/by/4.0/}{Creative Commons Attribution 4.0 International License}\medskip\\
       {Submitted: \submitteddate} \ifbool{isArticle}{ {\textbar} Accepted: \accepteddate {\textbar} Published: \publisheddate}{}\\
    \begin{center}
        \LARGE\textbf{\contribtitle}\\\smallskip
        \Large{\contribsubtitle}
    \end{center}
    \begin{center}
    \normalsize{\@author}
    \end{center}
    \begin{center}\small
        \affiliations
    \end{center}
    \textbf{\large Abstract:} {\large\thisabstract}\bigskip\\
    \textbf{\large Keywords:} {\large\keywords}
    \thispagestyle{plain}
    %disable the following commands for inclusion in the header
    \renewcommand{\orcid}[1]{}
    \renewcommand{\affOne}{}
    \renewcommand{\affTwo}{}
    \renewcommand{\affThree}{}
    \renewcommand{\affFour}{}
    \fontsize{12pt}{14pt}\selectfont
    }
\makeatother
\endinput
