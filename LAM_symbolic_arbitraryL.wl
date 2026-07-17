(* ::Package:: *)

(* 
  LAM_symbolic_arbitraryL.wl
  
  Symbolic routines for the replica-symmetric (RS) theory LAM sector,
  supporting arbitrary number of layers L.
  Ref: "Do Hopfield Networks Dream of Stored Patterns? 
  A Statistical--Mechanical Theory of Dreaming in Multidirectional Associative Memories" 
  (A. Barra, F. Durante, A. Ladiana, M. Marra Solazzo)
*)

BeginPackage["LAMsym`"];

LAMKernel::usage = "LAMKernel[gMat, Hvec, beta] returns the L x L RS LAM kernel C(H).";
LAMDet::usage = "LAMDet[gMat, Hvec, beta] returns det C(H).";
LAMLogDet::usage = "LAMLogDet[gMat, Hvec, beta] returns log det C(H).";
LAMLogDetGrad::usage = "LAMLogDetGrad[gMat, Hsym, beta] returns the log det C gradient via Jacobi's formula.";
LAMDquad::usage = "LAMDquad[gMat, Hsym, beta] returns the RS quadratic correction D_LAM(H).";
LAMDquadNum::usage = "LAMDquadNum[gMat, Hsym, beta] returns the numerator of the quadratic correction.";
LAMPiBar::usage = "LAMPiBar[gMat, Hsym, beta] returns the conjugate LAM overlaps Pibar_l.";
LAMFreeEnergyBlock::usage = "LAMFreeEnergyBlock[gMat, Hsym, beta, alpha] returns the RS LAM free-energy block.";
LAMPositivityCondition::usage = "LAMPositivityCondition[gMat, Hvec, beta] returns the eigenvalues of C(H).";
LAMHomogeneousSymbol::usage = "LAMHomogeneousSymbol[L, g, beta, H] builds the homogeneous symbolic kernel.";

Begin["`Private`"];

(* Utility to enforce a symmetric matrix with zero diagonal *)
symZeroDiag[gMat_?MatrixQ] := Module[{g = (gMat + Transpose[gMat])/2},
  g - DiagonalMatrix[Diagonal[g]]
];

(* LAM kernel C(H). 
   Square roots are kept separate to ensure automatic simplification 
   during determinant expansion without needing positivity assumptions. *)
LAMKernel[gMat_?MatrixQ, Hvec_List, beta_] := Module[{g = symZeroDiag[gMat], L = Length[Hvec]},
  Table[
    If[l === m, 1, - beta g[[l, m]] Sqrt[1 - Hvec[[l]]] Sqrt[1 - Hvec[[m]]]],
    {l, L}, {m, L}]
];

LAMDet[gMat_?MatrixQ, Hvec_List, beta_] := Det[LAMKernel[gMat, Hvec, beta]];
LAMLogDet[gMat_?MatrixQ, Hvec_List, beta_] := Log[LAMDet[gMat, Hvec, beta]];

(* Log-det gradient via Jacobi's formula. Hsym must be a list of symbols. *)
LAMLogDetGrad[gMat_?MatrixQ, Hsym_List, beta_] := Module[
  {C = LAMKernel[gMat, Hsym, beta], Ci, L = Length[Hsym]},
  Ci = Inverse[C];
  Table[Tr[Ci . D[C, Hsym[[l]]]], {l, L}]
];

(* RS quadratic correction *)
LAMDquad[gMat_?MatrixQ, Hsym_List, beta_] := Module[{grad = LAMLogDetGrad[gMat, Hsym, beta]},
  Sum[Hsym[[l]] grad[[l]], {l, Length[Hsym]}]
];

(* Numerator form of D_LAM *)
LAMDquadNum[gMat_?MatrixQ, Hsym_List, beta_] := Module[
  {detC = LAMDet[gMat, Hsym, beta], L = Length[Hsym]},
  (1/2) Sum[Hsym[[l]] D[detC, Hsym[[l]]], {l, L}]
];

(* Conjugate LAM overlaps Pibar_l *)
LAMPiBar[gMat_?MatrixQ, Hsym_List, beta_] := Module[
  {grad = LAMLogDetGrad[gMat, Hsym, beta], Dl, L = Length[Hsym]},
  Dl = Sum[Hsym[[m]] grad[[m]], {m, L}];
  Table[(2/beta) (grad[[l]] - D[Dl, Hsym[[l]]]), {l, L}]
];

(* RS LAM free-energy block *)
LAMFreeEnergyBlock[gMat_?MatrixQ, Hsym_List, beta_, alpha_] := Module[
  {ld = LAMLogDet[gMat, Hsym, beta], Dq = LAMDquad[gMat, Hsym, beta],
   pb = LAMPiBar[gMat, Hsym, beta], L = Length[Hsym]},
  - (alpha/2) ld + (alpha/2) Dq - (alpha beta/4) Sum[pb[[l]] (1 - Hsym[[l]]), {l, L}]
];

(* Positivity condition for the LAM Gaussian integral *)
LAMPositivityCondition[gMat_?MatrixQ, Hvec_List, beta_] := Eigenvalues[LAMKernel[gMat, Hvec, beta]];

(* Homogeneous symbolic kernel (all g_lm=g, all H_l=H) *)
LAMHomogeneousSymbol[L_Integer, g_, beta_, H_] := Module[{gMat = g (ConstantArray[1, {L, L}] - IdentityMatrix[L])},
  LAMKernel[gMat, ConstantArray[H, L], beta]
];

End[];
EndPackage[];


(* =====================================================================
   Checks & examples
   ===================================================================== *)

Module[{},
  Print["--- LAM symbolic routines validation (L=3) ---"];
  
  ClearAll[g, g12, g13, g23, beta, H, h1, h2, h3, u, q, alpha];

  Module[{C3h, detH, target},
    C3h = LAMsym`LAMHomogeneousSymbol[3, g, beta, H];
    detH = Simplify[Det[C3h]];
    target = Simplify[(1 - 2 u) (1 + u)^2 /. u -> beta g (1 - H)];
    Print["L=3 homogeneous det MATCH: ", Simplify[detH - target] === 0];
  ];

  Module[{gMat, C3, det3, target},
    gMat = {{0, g12, g13}, {g12, 0, g23}, {g13, g23, 0}};
    C3 = LAMsym`LAMKernel[gMat, {h1, h2, h3}, beta];
    det3 = Simplify[Det[C3]];
    target = Simplify[1 - beta^2 (g12^2 (1 - h1) (1 - h2) + g13^2 (1 - h1) (1 - h3) + g23^2 (1 - h2) (1 - h3)) - 2 beta^3 g12 g13 g23 (1 - h1) (1 - h2) (1 - h3)];
    Print["L=3 heterogeneous det MATCH: ", Simplify[det3 - target] === 0];
  ];

  Module[{gMat, grad, Cany, manual},
    gMat = {{0, g12, g13}, {g12, 0, g23}, {g13, g23, 0}};
    grad = LAMsym`LAMLogDetGrad[gMat, {h1, h2, h3}, beta];
    Cany = LAMsym`LAMKernel[gMat, {h1, h2, h3}, beta];
    manual = Simplify[Tr[Inverse[Cany] . D[Cany, h1]]];
    Print["Jacobi gradient (l=1) MATCH: ", Simplify[grad[[1]] - manual] === 0];
  ];

  Module[{Dnum, target},
    Dnum = LAMsym`LAMDquadNum[g (ConstantArray[1, {3, 3}] - IdentityMatrix[3]), {h1, h2, h3}, beta] /. {h1 -> q, h2 -> q, h3 -> q} // Simplify;
    target = Simplify[3 q (1 - q) beta^2 g^2 (1 + u) /. u -> beta g (1 - q)];
    Print["L=3 D_LAM^num MATCH: ", Simplify[Dnum - target] === 0];
  ];

  Module[{ev},
    ev = Simplify[LAMsym`LAMPositivityCondition[g (ConstantArray[1, {3, 3}] - IdentityMatrix[3]), {H, H, H}, beta]];
    Print["L=3 eigenvalues: ", ev];
  ];

  Print["\n--- General L tests ---"];
  
  Module[{C4, det4, pb4},
    C4 = LAMsym`LAMHomogeneousSymbol[4, g, beta, H];
    det4 = FullSimplify[Det[C4]];
    Print["L=4 homogeneous det MATCH: ", Simplify[det4 - ((1 - 3 u) (1 + u)^3 /. u -> beta g (1 - H))] === 0];
    
    pb4 = FullSimplify[LAMsym`LAMPiBar[g (ConstantArray[1, {4, 4}] - IdentityMatrix[4]), {h1, h2, h3, h4}, beta] /. {h1 -> H, h2 -> H, h3 -> H, h4 -> H}];
    Print["L=4 Pibar_l (homogeneous) = ", pb4[[1]]];
  ];

  Module[{ok},
    ok = Table[Simplify[Det[LAMsym`LAMHomogeneousSymbol[LL, g, beta, H]] - ((1 - (LL - 1) u) (1 + u)^(LL - 1) /. u -> beta g (1 - H))] === 0, {LL, 2, 6}];
    Print["L=2..6 det formula MATCH: ", And @@ ok];
  ];

  Module[{gMat, detF},
    gMat = N[0.3 (ConstantArray[1, {5, 5}] - IdentityMatrix[5])];
    detF = LAMsym`LAMDet[gMat, ConstantArray[H, 5], 2.0];
    Print["L=5 det C(H) at g=0.3, beta=2:"];
    Print["  H=0.0 -> ", detF /. H -> 0.0];
    Print["  H=0.5 -> ", detF /. H -> 0.5];
    Print["  H=0.9 -> ", detF /. H -> 0.9];
  ];
];

Module[{L = 4, gval = 0.4, betaval = 5.0, gMat, Hsym, piFun, Hval = 0.7},
  Print["\n--- Numerical closure example (L=4) ---"];
  gMat = gval (ConstantArray[1, {L, L}] - IdentityMatrix[L]);
  Hsym = Array[Symbol["hh" <> ToString[#]] &, L];
  piFun = LAMsym`LAMPiBar[gMat, Hsym, betaval];
  
  Print["det C (H=", Hval, ") = ", LAMsym`LAMDet[gMat, ConstantArray[Hval, L], betaval]];
  Print["D_LAM (H=", Hval, ") = ", N[LAMsym`LAMDquad[gMat, Hsym, betaval] /. Thread[Hsym -> ConstantArray[Hval, L]]]];
  Print["Pibar_1 (H=", Hval, ") = ", N[piFun[[1]] /. Thread[Hsym -> ConstantArray[Hval, L]]]];
];
