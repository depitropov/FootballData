#! /usr/bin/env python


# @timer
def first_half_goals(matches, side):
    if side == 'home':
        return matches.HTHG.sum() - matches.HTAG.sum()
    elif side == 'away':
        return matches.HTAG.sum() - matches.HTHG.sum()


def second_half_goals(matches, side):
    if side == 'home':
        return (matches.FTHG.sum() - matches.HTHG.sum()) - (matches.FTAG.sum() - matches.HTAG.sum())
    elif side == 'away':
        return (matches.FTAG.sum() - matches.HTAG.sum()) - (matches.FTHG.sum() - matches.HTHG.sum())


def half_wins(matches, side):
    if side == 'home':
        return matches[matches.HTR == 1].index.size
    elif side == 'away':
        return matches[matches.HTR == 2].index.size


def half_draws(matches, side):
    if side == 'home':
        return matches[matches.HTR == 3].index.size
    elif side == 'away':
        return matches[matches.HTR == 3].index.size


def half_lose(matches, side):
    if side == 'home':
        return matches[matches.HTR == 2].index.size
    elif side == 'away':
        return matches[matches.HTR == 1].index.size


def full_wins(matches, side):
    if side == 'home':
        return matches[matches.FTR == 1].index.size
    elif side == 'away':
        return matches[matches.FTR == 2].index.size


def full_draws(matches, side):
    if side == 'home':
        return matches[matches.FTR == 3].index.size
    elif side == 'away':
        return matches[matches.FTR == 3].index.size


def full_lose(matches, side):
    if side == 'home':
        return matches[matches.FTR == 2].index.size
    elif side == 'away':
        return matches[matches.FTR == 1].index.size


def win_odds(matches, side):
    if side == 'home':
        return matches.B365H.mean()
    if side == 'away':
        return matches.B365A.mean()


def draw_odds(matches, side):
    if side == 'home':
        return matches.B365D.mean()
    if side == 'away':
        return matches.B365D.mean()


def lose_odds(matches, side):
    if side == 'home':
        return matches.B365A.mean()
    if side == 'away':
        return matches.B365H.mean()


def home_draw(matches, side):
    if side == 'home':
        return matches[matches.HTFTR == 13].index.size
    if side == 'away':
        return matches[matches.HTFTR == 13].index.size


def away_draw(matches, side):
    if side == 'home':
        return matches[matches.HTFTR == 23].index.size
    if side == 'away':
        return matches[matches.HTFTR == 23].index.size


algorithms = (first_half_goals, second_half_goals, half_wins, half_draws, half_lose, full_wins,
              full_draws, full_lose, win_odds, draw_odds, lose_odds, home_draw, away_draw)

