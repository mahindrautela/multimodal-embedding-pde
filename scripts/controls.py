#!/usr/bin/env python3
"""Control inputs: notation-equivalent decks, heat-calibration decks,
mismatched deck pairing, time-reversed and first-frame contact sheets.

All rewrites are generated from the same resolved parameters as the primary
decks, so an equivalent deck specifies exactly the same mathematical problem.
"""
from __future__ import annotations

import re

import numpy as np

# Equivalent equations: diffusion written as second partials, terms reordered
# (diffusion last), and spatial coordinates renamed x -> xi, y -> zeta.
REWRITTEN_EQUATIONS = {
    "diffusion": (r"\partial_t u=D_u\left(u_{\xi\xi}+u_{\zeta\zeta}\right),\qquad "
                  r"\partial_t v=D_v\left(v_{\xi\xi}+v_{\zeta\zeta}\right)."),
    "advection_diffusion": (r"\partial_t u=-c_{\zeta} u_{\zeta}-c_{\xi} u_{\xi}+D_u\left(u_{\xi\xi}+u_{\zeta\zeta}\right),\qquad "
                            r"\partial_t v=-c_{\zeta} v_{\zeta}-c_{\xi} v_{\xi}+D_v\left(v_{\xi\xi}+v_{\zeta\zeta}\right)."),
    "linear_exchange": (r"\partial_t u=\kappa(v-u)+D_u\left(u_{\xi\xi}+u_{\zeta\zeta}\right),\qquad "
                        r"\partial_t v=\kappa(u-v)+D_v\left(v_{\xi\xi}+v_{\zeta\zeta}\right)."),
    "fisher_kpp": (r"\partial_t u=r\,u(1-u)+D_u\left(u_{\xi\xi}+u_{\zeta\zeta}\right),\qquad "
                   r"\partial_t v=r\,v(1-v)+D_v\left(v_{\xi\xi}+v_{\zeta\zeta}\right)."),
    "shifted_allen_cahn": (r"\partial_t u=a\,u(1-u)(u-\tfrac12)+D_u\left(u_{\xi\xi}+u_{\zeta\zeta}\right),\qquad "
                           r"\partial_t v=a\,v(1-v)(v-\tfrac12)+D_v\left(v_{\xi\xi}+v_{\zeta\zeta}\right)."),
    "gray_scott": (r"\partial_t u=F(1-u)-u v^2+D_u\left(u_{\xi\xi}+u_{\zeta\zeta}\right),\qquad "
                   r"\partial_t v=u v^2-(F+k)v+D_v\left(v_{\xi\xi}+v_{\zeta\zeta}\right)."),
}
REWRITTEN_DOMAIN = (r"\Omega=[0,1]^2,\qquad U=(u,v)^\top.\quad "
                    r"\xi_p=(p+\tfrac12)/64,\ \zeta_q=(q+\tfrac12)/64,\ p,q=0,\ldots,63.")

_X = re.compile(r"(?<![A-Za-z\\])x(?![A-Za-z])")
_Y = re.compile(r"(?<![A-Za-z\\])y(?![A-Za-z])")


def rename_coordinates(latex: str) -> str:
    """Rename bound spatial coordinates x->xi, y->zeta in IC and BC formulas."""
    latex = latex.replace(r"\partial_x", r"\partial_{\xi}").replace(r"\partial_y", r"\partial_{\zeta}")
    return _Y.sub(r"{\\zeta}", _X.sub(r"{\\xi}", latex))


def split_sections(text: str) -> dict[str, str]:
    sections = {}
    for block in text.strip("\n").split("\n\n"):
        head, _, body = block.partition(":\n")
        sections[head] = body
    return sections


def equivalent_deck(text: str, pde: str) -> str:
    s = split_sections(text)
    assert list(s) == ["Equation", "Parameters", "Initial condition", "Boundary", "Domain", "Time"], list(s)
    ic_lines = s["Initial condition"].split("\n")
    ic = "\n".join([ic_lines[0]] + [rename_coordinates(line) for line in ic_lines[1:]])  # eta(s) unchanged
    out = {
        "Equation": REWRITTEN_EQUATIONS[pde],
        "Parameters": s["Parameters"].replace("c_x=", r"c_{\xi}=").replace("c_y=", r"c_{\zeta}="),
        "Initial condition": ic,
        "Boundary": rename_coordinates(s["Boundary"]),
        "Domain": REWRITTEN_DOMAIN,
        "Time": s["Time"],
    }
    return "\n\n".join(f"{k}:\n{v}" for k, v in out.items()) + "\n"


HEAT_ANCHOR = 0.01
HEAT_CANDIDATES = [0.011, 0.02, 0.05]


def heat_deck(nu: float) -> str:
    sections = {
        "Equation": r"\partial_t u=\nu\Delta u.",
        "Parameters": rf"\nu={nu:.8f}.",
        "Initial condition": r"u(0,x,y)=\sin(\pi x)\sin(\pi y).",
        "Boundary": r"u(t,x,y)=0\quad\text{for }(x,y)\in\partial\Omega.",
        "Domain": r"\Omega=[0,1]^2,\qquad \Delta=\partial_{xx}+\partial_{yy}.",
        "Time": (r"0\leq t\leq1,\qquad t_n=n/99,\quad n=0,\ldots,99.\quad "
                 r"\text{Observed frames: }n\in\{0,25,50,74,99\}."),
    }
    return "\n\n".join(f"{k}:\n{v}" for k, v in sections.items()) + "\n"


def heat_rms(nu_a: float, nu_b: float, frames=(0, 25, 50, 74, 99)) -> float:
    t = np.asarray(frames) / 99.0
    d = np.exp(-2 * np.pi**2 * nu_a * t) - np.exp(-2 * np.pi**2 * nu_b * t)
    return float(0.5 * np.sqrt(np.mean(d**2)))


def mismatch_partner(pde_index: int, n_pdes: int = 6) -> int:
    """Deck of the next PDE class (cyclic), same IC family, BC and realization."""
    return (pde_index + 1) % n_pdes


TIME_REVERSAL = [4, 3, 2, 1, 0]  # panels reversed; header labels stay t=0.00 ... t=1.00
