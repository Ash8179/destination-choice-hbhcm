# =============================================================================
# E.1 - SP CHOICE EXPERIMENT DESIGN PIPELINE
# D-Efficient Design for Destination Choice Study
# =============================================================================
#
# PIPELINE OVERVIEW
# -----------------
# Stage 0 : Package loading and environment setup
# Stage 1 : Attitudinal LV correlation matrix — visualise and diagnose
# Stage 2 : Attribute levels, effects coding, prohibited combinations
# Stage 3 : Pre-filter candidate set, generate D-efficient design (Modfed)
# Stage 4 : Design matrix validation (correlations, VIF, level balance, dominance)
# Stage 5 : Construct validity check sets (extreme profiles + test-retest)
# Stage 6 : Merge, block assignment, and CSV export
# Stage 7 : [Optional] DB-optimal upgrade using pilot priors
#
# DESIGN STRUCTURE
# ----------------
#   Main design      : 24 D-efficient choice sets
#   Extreme VC sets  : 4 hand-crafted sets (1 per block)
#                      Alt1 clearly dominates Alt2; screens low-engagement respondents
#   Test-retest sets : 4 repeated sets (1 per block, repeat of set 3/9/15/21)
#                      Assesses within-respondent consistency
#   Total            : 32 choice set slots across 4 blocks
#   Per respondent   : 8 sets (6 main + 1 extreme VC + 1 test-retest)
#   Each set         : Alt1 + Alt2 + Alt3/Opt-Out (3-alternative choice)
#
# MODFED CONFIGURATION (confirmed working)
#   n.alts    = 3            — three alternatives per set drawn from candidate set
#   alt.cte   = c(0, 0, 1)  — Alt1/Alt2: no ASC; Alt3: ASC (acts as opt-out)
#   no.choice = FALSE        — opt-out handled as Alt3 with ASC, not special row
#   Row name pattern: "set*.alt1", "set*.alt2", "set*.alt3"
#   Dimensions: (n.sets * 3) rows x 17 cols (1 ASC col + 16 attribute cols)
#
# BLOCK STRUCTURE
#   Block 1: Main sets  1-6  | Extreme VC set 25 | Test-retest set 29 (= set  3)
#   Block 2: Main sets  7-12 | Extreme VC set 26 | Test-retest set 30 (= set  9)
#   Block 3: Main sets 13-18 | Extreme VC set 27 | Test-retest set 31 (= set 15)
#   Block 4: Main sets 19-24 | Extreme VC set 28 | Test-retest set 32 (= set 21)
#
# ATTRIBUTES (0-indexed in output; 1-indexed internally)
#   Trans_L : Transport increment  — 5 levels (0-4)
#   Dest_L  : Destination quality  — 3 levels (0-2)
#   Vib_L   : Vibrancy             — 3 levels (0-2)
#   Plea_L  : Pleasantness         — 3 levels (0-2)
#   Walk_L  : Walkability          — 3 levels (0-2)
#   Safe_L  : Safety               — 3 levels (0-2)
#   Exp_L   : Experiential         — 3 levels (0-2)
#
# PROHIBITED COMBINATIONS (from attitudinal LV correlations)
#   Vib=High(3) + Plea=High(3) : implausibly ideal in Singapore context
#   Vib=Low(1)  + Plea=Low(1)  : trivially unattractive
#   Walk=High(3) + Safe=Low(1) : inconsistent with Singapore pedestrian context
#
# RESPONDENT QUALITY SCREENING (post-collection)
#   Flag   : fail extreme VC OR inconsistent test-retest
#   Remove : fail extreme VC AND inconsistent test-retest
#
#   Author: Zhang Wenyu
#   Date: 2026-03-14
# =============================================================================


# =============================================================================
# STAGE 0: Package loading
# =============================================================================

required_packages <- c("idefix", "corrplot", "car", "tidyverse", "MASS")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}
set.seed(2025)


# =============================================================================
# STAGE 1: Attitudinal LV correlation matrix
# =============================================================================

lv_cor <- matrix(c(
  1.000000, -0.690875, -0.259394,  0.143215,  0.169025,
  -0.690875,  1.000000,  0.422666, -0.099436,  0.233315,
  -0.259394,  0.422666,  1.000000, -0.380700,  0.045295,
  0.143215, -0.099436, -0.380700,  1.000000, -0.057397,
  0.169025,  0.233315,  0.045295, -0.057397,  1.000000
), nrow = 5, byrow = TRUE)

lv_names <- c("Vibrancy", "Pleasantness", "Walkability", "Safety", "Experiential")
colnames(lv_cor) <- rownames(lv_cor) <- lv_names

cat("=== Attitudinal LV Correlation Matrix ===\n")
print(round(lv_cor, 3))

cat("\n=== Pairs Exceeding |r| > 0.35 (design risk) ===\n")
cor_long <- as.data.frame(as.table(lv_cor)) %>%
  filter(as.character(Var1) < as.character(Var2)) %>%
  rename(LV1 = Var1, LV2 = Var2, r = Freq) %>%
  filter(abs(r) > 0.35) %>%
  arrange(desc(abs(r)))
print(cor_long)

png("design_corr_lv.png", width = 600, height = 520, res = 100)
corrplot(lv_cor,
         method = "color", type = "upper", addCoef.col = "black",
         number.cex = 0.85, tl.col = "black", tl.srt = 45,
         col = colorRampPalette(c("#D73027", "white", "#1A9850"))(200),
         title = "Attitudinal LV Correlation Matrix", mar = c(0, 0, 2, 0))
dev.off()
cat("Saved: design_corr_lv.png\n\n")


# =============================================================================
# STAGE 2: Attribute specification and prohibited combinations
# =============================================================================

lvls       <- c(5, 3, 3, 3, 3, 3, 3)
n_attr     <- length(lvls)
attr_names <- c("Trans_L", "Dest_L", "Vib_L", "Plea_L", "Walk_L", "Safe_L", "Exp_L")
coding     <- rep("E", n_attr)

prohibited <- rbind(
  c(NA, NA, 3, 3, NA, NA, NA),   # Vib=High + Plea=High
  c(NA, NA, 1, 1, NA, NA, NA),   # Vib=Low  + Plea=Low
  c(NA, NA, NA, NA, 3, 1, NA)    # Walk=High + Safe=Low
)

cat("=== Prohibited Combinations (1-indexed) ===\n")
cat("Columns: Trans_L  Dest_L  Vib_L  Plea_L  Walk_L  Safe_L  Exp_L\n")
print(prohibited)
cat("\n")


# =============================================================================
# STAGE 3: Pre-filter candidate set and generate D-efficient design
# =============================================================================

cand_set <- Profiles(lvls = lvls, coding = coding)
cat(sprintf("Candidate profiles before filtering: %d\n", nrow(cand_set)))

# Decode effects-coded candidate set to 1-indexed level integers
# idefix Profiles() uses 0/1 dummy-style effects coding:
#   Level j (j = 1..k-1): column j = 1, all other columns = 0
#   Level k (reference) : all columns = 0
decode_cand <- function(cand, lvls) {
  out       <- matrix(NA_integer_, nrow = nrow(cand), ncol = length(lvls))
  col_start <- 1
  for (i in seq_along(lvls)) {
    k   <- lvls[i]
    idx <- col_start:(col_start + k - 2)
    blk <- cand[, idx, drop = FALSE]
    out[, i] <- apply(blk, 1, function(row) {
      pos <- which(row > 0.5)
      if (length(pos) == 1L) pos else k
    })
    col_start <- col_start + k - 1
  }
  out
}

cand_decoded <- decode_cand(cand_set, lvls)

cat("\nDecoded level ranges:\n")
for (i in seq_along(attr_names)) {
  cat(sprintf("  %-8s: %d - %d\n", attr_names[i],
              min(cand_decoded[, i]), max(cand_decoded[, i])))
}

is_prohibited <- function(profile_mat, prohib_mat) {
  apply(profile_mat, 1, function(prof) {
    any(apply(prohib_mat, 1, function(rule) {
      non_na <- !is.na(rule)
      all(prof[non_na] == rule[non_na])
    }))
  })
}

exclude_mask  <- is_prohibited(cand_decoded, prohibited)
cand_filtered <- cand_set[!exclude_mask, , drop = FALSE]
cat(sprintf("\nProfiles removed : %d\n", sum(exclude_mask)))
cat(sprintf("Profiles remaining: %d\n\n", nrow(cand_filtered)))

# par.draws structure for n.alts=3, alt.cte=c(0,0,1):
#   [[1]] ncol = n.cte = 1 (one non-zero in alt.cte) -> ASC draws
#   [[2]] ncol = ncol(cand.set) = 16                 -> attribute draws
n_asc      <- sum(c(0, 0, 1) != 0)   # = 1
n_attr_par <- ncol(cand_filtered)     # = 16

zero_prior <- list(
  matrix(0, nrow = 1, ncol = n_asc),
  matrix(0, nrow = 1, ncol = n_attr_par)
)

cat("=== Generating D-Efficient Design: 24 main choice sets ===\n")
cat("(3 alternatives per set: Alt1, Alt2, Alt3/Opt-out; may take 3-5 minutes)\n")
set.seed(2025)
design_output <- Modfed(
  cand.set  = cand_filtered,
  n.sets    = 24,
  n.alts    = 3,             # Alt1 + Alt2 + Alt3(opt-out)
  alt.cte   = c(0, 0, 1),   # Alt1: no ASC | Alt2: no ASC | Alt3: ASC (opt-out)
  par.draws = zero_prior,
  no.choice = FALSE,         # opt-out is Alt3 with ASC, not a special no.choice row
  parallel  = FALSE
)

best       <- design_output$BestDesign
design_mat <- best$design

cat("\n=== Design Generation Complete ===\n")
cat(sprintf("DB-error     : %.6f\n", best$DB.error))
cat(sprintf("AB-error     : %.6f\n", best$AB.error))
cat(sprintf("Orthogonality: %.4f\n\n", best$Orthogonality))


# =============================================================================
# STAGE 4: Design matrix validation
# =============================================================================

# Row name pattern confirmed: "set*.alt1", "set*.alt2", "set*.alt3"
# 72 rows = 24 sets x 3 alts; 17 cols = 1 ASC col + 16 attribute cols
alt1_rows   <- grepl("alt1", rownames(design_mat))
alt2_rows   <- grepl("alt2", rownames(design_mat))
alt3_rows   <- grepl("alt3", rownames(design_mat))

design_alt1 <- design_mat[alt1_rows, , drop = FALSE]
design_alt2 <- design_mat[alt2_rows, , drop = FALSE]
design_alt3 <- design_mat[alt3_rows, , drop = FALSE]   # opt-out: all zeros

cat(sprintf("Alt1 rows: %d | Alt2 rows: %d | Alt3(opt-out) rows: %d\n\n",
            nrow(design_alt1), nrow(design_alt2), nrow(design_alt3)))

cat("Column names:\n")
print(colnames(design_mat))

# Identify attribute columns (exclude ASC column named "*cte*")
attr_col_idx <- which(!grepl("cte", colnames(design_mat)))
cat(sprintf("\nAttribute columns (%d): %s\n\n",
            length(attr_col_idx),
            paste(colnames(design_mat)[attr_col_idx], collapse = " ")))

# Map column ranges per attribute
col_ranges <- vector("list", n_attr)
col_ptr    <- 1
for (i in seq_along(lvls)) {
  col_ranges[[i]] <- attr_col_idx[col_ptr:(col_ptr + lvls[i] - 2)]
  col_ptr         <- col_ptr + lvls[i] - 1
}

cat("Column mapping per attribute:\n")
for (i in seq_along(attr_names)) {
  cat(sprintf("  %-8s: [%s]\n", attr_names[i],
              paste(colnames(design_mat)[col_ranges[[i]]], collapse = " ")))
}

# Decode Alt1 and Alt2 rows (Alt3 is opt-out, all-zero, not decoded)
decode_design_rows <- function(rows, col_ranges, lvls, attr_names) {
  mat <- matrix(NA_integer_, nrow = nrow(rows), ncol = length(lvls))
  colnames(mat) <- attr_names
  for (i in seq_along(lvls)) {
    blk      <- rows[, col_ranges[[i]], drop = FALSE]
    mat[, i] <- apply(blk, 1, function(row) {
      pos <- which(row > 0.5)
      if (length(pos) == 1L) pos else lvls[i]
    })
  }
  mat
}

dec_alt1 <- decode_design_rows(design_alt1, col_ranges, lvls, attr_names)
dec_alt2 <- decode_design_rows(design_alt2, col_ranges, lvls, attr_names)

cat("\n=== Decoded Alt1 (first 8 rows) ===\n")
print(head(dec_alt1, 8))
cat("\n=== Decoded Alt2 (first 8 rows) ===\n")
print(head(dec_alt2, 8))

# Correlation check (Alt1 + Alt2 pooled, n=48)
dec_all    <- rbind(dec_alt1, dec_alt2)
cor_design <- cor(dec_all)

cat("\n=== Design Matrix Attribute Correlations (Alt1+Alt2 pooled, n=48) ===\n")
print(round(cor_design, 3))

flagged <- FALSE
for (i in 1:(n_attr - 1)) {
  for (j in (i + 1):n_attr) {
    if (abs(cor_design[i, j]) > 0.2) {
      if (!flagged) { cat("\nWARNING — correlations > 0.2:\n"); flagged <- TRUE }
      cat(sprintf("  %s x %s : r = %.3f\n",
                  attr_names[i], attr_names[j], cor_design[i, j]))
    }
  }
}
if (!flagged) cat("OK — all correlations < 0.2\n")

png("design_corr_matrix.png", width = 600, height = 520, res = 100)
corrplot(cor_design,
         method = "color", type = "upper", addCoef.col = "black",
         number.cex = 0.8, tl.col = "black", tl.srt = 45,
         col = colorRampPalette(c("#D73027", "white", "#1A9850"))(200),
         title = "Design Matrix Correlation (Alt1+Alt2 pooled)", mar = c(0, 0, 2, 0))
dev.off()
cat("Saved: design_corr_matrix.png\n\n")

# VIF check
cat("=== VIF Check ===\n")
decoded_df <- as.data.frame(dec_all)
vif_vals   <- vif(lm(Vib_L ~ Plea_L + Walk_L + Safe_L + Exp_L + Trans_L + Dest_L,
                     data = decoded_df))
print(round(vif_vals, 3))
if (any(vif_vals > 5)) {
  cat("WARNING — VIF > 5.\n")
} else if (any(vif_vals > 3)) {
  cat("CAUTION — VIF 3-5.\n")
} else {
  cat("OK — all VIF < 3.\n")
}

# Level balance
cat("\n=== Level Frequency Balance (Alt1+Alt2 pooled, n=48) ===\n")
for (i in seq_along(attr_names)) {
  freq  <- table(dec_all[, i])
  ideal <- 48 / lvls[i]
  cat(sprintf("%-10s: %s  (ideal=%.1f)\n", attr_names[i],
              paste(names(freq), freq, sep = "=", collapse = "  "), ideal))
}

# Dominance check: flag sets where one alt is >= other on ALL attributes
cat("\n=== Dominance Check (Alt1 vs Alt2) ===\n")
dominant_sets <- c()
for (s in seq_len(24)) {
  a1 <- dec_alt1[s, ]
  a2 <- dec_alt2[s, ]
  if ((all(a1 >= a2) && any(a1 > a2)) || (all(a2 >= a1) && any(a2 > a1))) {
    dominant_sets <- c(dominant_sets, s)
    cat(sprintf("  Set %02d: one alternative dominates\n", s))
  }
}
if (length(dominant_sets) == 0) {
  cat("OK — No dominant alternatives in main design.\n\n")
} else {
  cat(sprintf("WARNING — %d dominant set(s): %s\n\n",
              length(dominant_sets), paste(dominant_sets, collapse = ", ")))
}


# =============================================================================
# STAGE 5: Validity check sets
# =============================================================================

# --- 5a. Extreme VC sets (4 sets, one per block) ---
# Alt1 clearly dominates Alt2 on the most decision-relevant attributes.
# Rational respondents should choose Alt1. Used post-hoc for quality screening.
# All levels 0-indexed below. Prohibition rules verified in check loop.
#
# Prohibition rules (0-indexed equivalent):
#   Vib=2 + Plea=2 : both High     -> prohibited
#   Vib=0 + Plea=0 : both Low      -> prohibited
#   Walk=2 + Safe=0: Walk High + Safe Low -> prohibited

extreme_vc <- data.frame(
  VC_ID   = rep(1:4, each = 2),
  Alt_ID  = rep(c("Alt1", "Alt2"), 4),
  Trans_L = c(0, 3,   # VC1: low vs high cost
              1, 4,   # VC2: moderate vs very high
              0, 4,   # VC3: lowest vs highest
              1, 3),  # VC4: low vs high
  Dest_L  = c(2, 0,
              2, 0,
              1, 0,
              2, 0),
  Vib_L   = c(1, 1,   # VC1: both mid (Alt2 low Plea -> ok; avoids Low+Low)
              0, 1,   # VC2: Alt1 low vib, Alt2 mid
              1, 1,   # VC3: both mid (avoids Low+Low for Alt2)
              1, 0),  # VC4: Alt1 mid, Alt2 low (Plea=mid for Alt2 -> ok)
  Plea_L  = c(2, 1,   # VC1: Alt1 high, Alt2 mid (Vib=mid+Plea=mid: ok)
              2, 0,   # VC2: Alt1 high (Vib=low -> ok), Alt2 low (Vib=mid -> ok)
              2, 1,   # VC3: Alt1 high, Alt2 mid (avoids Low+Low)
              2, 1),  # VC4: Alt1 high, Alt2 mid (Vib=low+Plea=mid: ok)
  Walk_L  = c(2, 0,
              1, 0,
              2, 0,
              2, 0),
  Safe_L  = c(2, 1,   # VC1: Alt2 Safe=mid (Walk=0 -> no Walk-High+Safe-Low issue)
              2, 0,   # VC2: Alt2 Walk=0 -> Safe=Low ok
              2, 1,   # VC3: Alt2 Walk=0 -> Safe=mid for extra contrast
              2, 0),  # VC4: Alt2 Walk=0 -> Safe=Low ok
  Exp_L   = c(1, 0,
              2, 0,
              1, 0,
              2, 0)
)

# Prohibition check for VC sets
cat("=== Extreme VC Sets — Prohibition Check ===\n")
violation_found <- FALSE
for (r in 1:nrow(extreme_vc)) {
  v <- extreme_vc[r, "Vib_L"]  + 1L
  p <- extreme_vc[r, "Plea_L"] + 1L
  w <- extreme_vc[r, "Walk_L"] + 1L
  s <- extreme_vc[r, "Safe_L"] + 1L
  msg <- ""
  if (v == 3 && p == 3) msg <- "Vib=High+Plea=High"
  if (v == 1 && p == 1) msg <- "Vib=Low+Plea=Low"
  if (w == 3 && s == 1) msg <- "Walk=High+Safe=Low"
  if (msg != "") {
    cat(sprintf("  VIOLATION in VC%d %s: %s\n",
                extreme_vc[r, "VC_ID"], extreme_vc[r, "Alt_ID"], msg))
    violation_found <- TRUE
  }
}
if (!violation_found) cat("  No violations. All VC profiles are valid.\n\n")

cat("Extreme VC sets (0-indexed):\n")
print(extreme_vc)

# --- 5b. Test-retest sets ---
# Exact repeats of main sets 3, 9, 15, 21 (one from each block).
# Position within block: set 3 is the 3rd of 6 main sets — central, avoids
# primacy/recency effects. Inconsistency rate > 25% flags low engagement.

retest_source <- c(3, 9, 15, 21)
retest_alt1   <- dec_alt1[retest_source, , drop = FALSE]
retest_alt2   <- dec_alt2[retest_source, , drop = FALSE]

cat(sprintf("\n=== Test-Retest Sets: repeating main sets %s ===\n",
            paste(retest_source, collapse = ", ")))
cat("Alt1 profiles:\n"); print(retest_alt1 - 1L)
cat("Alt2 profiles:\n"); print(retest_alt2 - 1L)


# =============================================================================
# STAGE 6: Merge, block assignment, and CSV export
# =============================================================================

# --- 6a. Main design (0-indexed, 3 rows per set: Alt1, Alt2, Opt_Out) ---
main_df <- bind_rows(
  data.frame(Set_Type = "Main", Choice_Set = seq_len(24),
             Alt_ID = "Alt1", as.data.frame(dec_alt1 - 1L)),
  data.frame(Set_Type = "Main", Choice_Set = seq_len(24),
             Alt_ID = "Alt2", as.data.frame(dec_alt2 - 1L)),
  data.frame(Set_Type = "Main", Choice_Set = seq_len(24),
             Alt_ID = "Opt_Out",
             as.data.frame(matrix(NA_integer_, 24, n_attr,
                                  dimnames = list(NULL, attr_names))))
) %>% arrange(Choice_Set, Alt_ID)

# --- 6b. Extreme VC sets (Choice_Set 25-28) ---
vc_extreme_alts <- extreme_vc %>%
  mutate(Set_Type   = "Extreme_VC",
         Choice_Set = VC_ID + 24L) %>%
  dplyr::select(Set_Type, Choice_Set, Alt_ID, all_of(attr_names))

vc_extreme_opt <- data.frame(
  Set_Type   = "Extreme_VC",
  Choice_Set = 25:28,
  Alt_ID     = "Opt_Out",
  as.data.frame(matrix(NA_integer_, 4, n_attr, dimnames = list(NULL, attr_names)))
)

vc_extreme_df <- bind_rows(vc_extreme_alts, vc_extreme_opt) %>%
  arrange(Choice_Set, Alt_ID)

# --- 6c. Test-retest sets (Choice_Set 29-32) ---
vc_retest_df <- bind_rows(
  data.frame(Set_Type = "Test_Retest", Choice_Set = 29:32, Alt_ID = "Alt1",
             Retest_Of = retest_source,
             as.data.frame(retest_alt1 - 1L)),
  data.frame(Set_Type = "Test_Retest", Choice_Set = 29:32, Alt_ID = "Alt2",
             Retest_Of = retest_source,
             as.data.frame(retest_alt2 - 1L)),
  data.frame(Set_Type = "Test_Retest", Choice_Set = 29:32, Alt_ID = "Opt_Out",
             Retest_Of = retest_source,
             as.data.frame(matrix(NA_integer_, 4, n_attr,
                                  dimnames = list(NULL, attr_names))))
) %>% arrange(Choice_Set, Alt_ID)

# --- 6d. Add Retest_Of column to main and VC (NA) ---
main_df$Retest_Of       <- NA_integer_
vc_extreme_df$Retest_Of <- NA_integer_

# --- 6e. Block assignment ---
# Each respondent answers exactly one block (8 sets):
#   6 main sets + 1 extreme VC set + 1 test-retest set
block_map <- data.frame(
  Choice_Set = c(1:6,  7:12, 13:18, 19:24,  # main
                 25,   26,   27,   28,        # extreme VC
                 29,   30,   31,   32),       # test-retest
  Block      = c(rep(1,6), rep(2,6), rep(3,6), rep(4,6),
                 1, 2, 3, 4,
                 1, 2, 3, 4)
)

full_design <- bind_rows(main_df, vc_extreme_df, vc_retest_df) %>%
  left_join(block_map, by = "Choice_Set") %>%
  dplyr::select(Block, Set_Type, Choice_Set, Retest_Of,
                Alt_ID, all_of(attr_names)) %>%
  arrange(Block, Choice_Set, Alt_ID)

# --- 6f. Export ---
write.csv(full_design, "sp_design_32sets.csv", row.names = FALSE, na = "")
cat("\nSaved: sp_design_32sets.csv\n")

block_summary <- full_design %>%
  filter(Alt_ID == "Alt1") %>%
  group_by(Block) %>%
  summarise(
    Total_sets  = n(),
    Main_sets   = paste(Choice_Set[Set_Type == "Main"],        collapse = ", "),
    Extreme_VC  = paste(Choice_Set[Set_Type == "Extreme_VC"],  collapse = ", "),
    Retest_set  = paste(Choice_Set[Set_Type == "Test_Retest"], collapse = ", "),
    Retest_of   = paste(Retest_Of[Set_Type == "Test_Retest"],  collapse = ", "),
    .groups = "drop"
  )
write.csv(block_summary, "sp_block_summary.csv", row.names = FALSE)
cat("Saved: sp_block_summary.csv\n\n")

cat("=== Block Summary ===\n")
print(block_summary)

cat("\n=== Full Block 1 Preview ===\n")
print(full_design %>% filter(Block == 1) %>% as.data.frame())


# =============================================================================
# STAGE 7: DB-optimal upgrade after pilot data collection
# =============================================================================
# After pilot (n ~ 20-30), re-run with Bayesian priors for better efficiency.
# Only re-run the 24 main sets; VC and retest sets remain unchanged.
#
# library(mlogit)
# pilot <- read.csv("pilot_responses.csv")
# pilot_long <- mlogit.data(pilot, choice="choice", shape="long",
#                            alt.var="alt_id", id.var="resp_id")
# mnl_pilot  <- mlogit(choice ~ Trans_L + Dest_L + Vib_L + Plea_L +
#                                Walk_L + Safe_L + Exp_L | 0,
#                      data = pilot_long)
# pilot_coef <- coef(mnl_pilot)
# pilot_vcov <- vcov(mnl_pilot)
# attr_draws <- MASS::mvrnorm(200, mu = pilot_coef, Sigma = pilot_vcov)
# asc_draws  <- matrix(rnorm(200, 0, 0.5), nrow = 200, ncol = 1)
# db_prior   <- list(asc_draws, attr_draws)   # [[1]]=ASC, [[2]]=attrs
#
# set.seed(2025)
# design_db <- Modfed(
#   cand.set  = cand_filtered, n.sets = 24, n.alts = 3,
#   alt.cte   = c(0, 0, 1), par.draws = db_prior,
#   no.choice = FALSE, parallel = FALSE
# )
# cat("DB-optimal:", round(design_db$BestDesign$DB.error, 6), "\n")
# cat("Zero-prior:", round(design_output$BestDesign$DB.error, 6), "\n")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

cat("\n============================================================\n")
cat("  SP DESIGN COMPLETE\n")
cat("============================================================\n")
cat(sprintf("  Main choice sets (D-efficient) : 24\n"))
cat(sprintf("  Extreme VC sets (hand-crafted) : 4\n"))
cat(sprintf("  Test-retest sets               : 4\n"))
cat(sprintf("  Total choice set slots         : 32\n"))
cat(sprintf("  Blocks                         : 4\n"))
cat(sprintf("  Sets per respondent            : 8 (6 main + 1 VC + 1 retest)\n"))
cat(sprintf("  Alternatives per set           : Alt1 + Alt2 + Opt_Out\n"))
cat(sprintf("  DB-error (main, zero priors)   : %.6f\n", best$DB.error))
cat(sprintf("  AB-error                       : %.6f\n", best$AB.error))
cat(sprintf("  Orthogonality                  : %.4f\n", best$Orthogonality))
cat(sprintf("  Profiles removed (prohib)      : %d / %d\n",
            sum(exclude_mask), nrow(cand_set)))
cat(sprintf("  Dominant sets in main design   : %d\n", length(dominant_sets)))
cat("------------------------------------------------------------\n")
cat("  Quality screening (post-collection):\n")
cat("    Flag   : fail extreme VC OR inconsistent retest\n")
cat("    Remove : fail extreme VC AND inconsistent retest\n")
cat("------------------------------------------------------------\n")
cat("  Output files:\n")
cat("    sp_design_32sets.csv   — full design with blocks\n")
cat("    sp_block_summary.csv   — block assignment table\n")
cat("    design_corr_lv.png     — attitudinal LV correlations\n")
cat("    design_corr_matrix.png — design matrix correlations\n")
cat("============================================================\n")
