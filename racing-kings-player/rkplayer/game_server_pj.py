import json
import requests
import sseclient
import rkplayer.game as g
import rkplayer.game_generator
import rkplayer.neural_network


'''
TODO:
-Update changes in API (settings, state and personal timebudget)
-create dictionary to hand over to NN with all relevant data

'''

# server URL
API_ENDPOINT = "https://pjki.ml/api"

# start position racingKings
START_POSITION = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"

# timeBudget and timeout set from PJKI -Team
TIME_BUDGET = 120000
TIME_OUT = 60000


# ---------------------- Wrapper functions --------------------- #


def new_setup(team_name, player_name, isis_name='', game='racingKings'):
    """Set up new team and player.

    isis_name is optional
    racingKings is default for game

    """
    team_id, team_token = create_team(team_name, isis_name, game)
    player_id, player_token = create_player(player_name, team_token)

    return team_id, team_token, player_id, player_token


def run_game(game_id, player, network):
    """Set ups game information and player_info."""
    game_name, game_type, players, settings, state = get_game(game_id)
    if game_type != "racingKings":
        print("game_type is not racingKings")
        return "error"

    own_player_id, own_player_token = extract_player_info(player)
    own_player_color, own_player_nr = player_mapping(own_player_id, players)
    own_player_time_budget = player_time_budget(own_player_id, state)

    a0_game = g.Game(state['fen'])

    player_info = ({'player_id': own_player_id,
                    'player_token': own_player_token,
                    'player_nr': own_player_nr,
                    'player_color': own_player_color,
                    'player_time_budget': own_player_time_budget})


    print("Start game: " + str(game_name))
    print("Settings: " + str(settings))
    print(a0_game)
    winner = start_game(game_id, player_info, a0_game, network)
    return winner


def start_game(game_id, player_info, a0_game, network):
    """Start new game with all information."""
    first_turn = False
    if player_info['player_nr'] == 'playerA':
        first_turn = True

    if first_turn:
        valid = apply_move(game_id, player_info, a0_game, network)
        if valid is not True:
            return "not valid"

    winner = get_events(game_id, player_info, a0_game, network)
    return winner


def apply_move(game_id, player_info, a0_game, network):
    """Get move from NN and apply it to own game & game_server."""
    action = 'move'
    if network == 'NULL':
        move = a0_game.random_move()
        move_id = a0_game.id_from_move(move)
    else:
        move_id, _ = rkplayer.game_generator.find_move(a0_game, network)
        move = a0_game.move_from_id(move_id)
    valid = send_event(game_id, player_info['player_token'], action, move)
    if valid is True:
        a0_game.make_move(move_id)
    return valid


# ----------------------- Endpoint functions --------------------- #


def create_team(team_name, isis_name='', game_type='racingKings'):
    """Create new team.

    isis_name is optional

    Returns
    team_id & team_token

    """
    data = {'name': team_name, 'isisName': isis_name, 'type': game_type}
    request_url = str(API_ENDPOINT) + str('/teams')

    response = requests.post(request_url, json=data)
    if response.status_code != 201:
        print("Fehler POST create_team")
        print(response.content)
        return "default", "default"

    response_data = response.json()
    team_id = response_data['id']
    team_token = response_data['token']

    print("team_id: " + str(team_id))
    print("team_token: " + str(team_token))
    return team_id, team_token


def test_team_token(team_token):
    """Test team_token.

    Returns
    team_id & valid boolean

    """
    valid = False
    token = "Basic " + str(team_token)
    auth = {'Authorization': token}
    request_url = str(API_ENDPOINT) + str('/teamlogin')

    response = requests.get(request_url, headers=auth)
    if response.status_code != 200:
        print("Fehler GET test_team_token")
        print(response.content)
        return "error", valid

    response_data = response.json()
    team_id = response_data['id']
    valid = response_data['valid']

    print("team_id: " + str(team_id))
    print("valid: " + str(valid))
    return team_id, valid


def list_teams(count=100, start=0):
    """List all existing teams.

    count and start limits the list received

    Returns
    list of teams

    """
    request_url = (str(API_ENDPOINT) + str('/teams?count=') +
                   str(count) + str('&start=') +
                   str(start))

    response = requests.get(request_url)
    if response.status_code != 200:
        print("Fehler GET list_team")
        print(response.content)
        return "error"

    teams = response.json()

    print(teams)
    return teams


def get_team(team_id):
    """Get team by team_id.

    Returns
    team_name, isis_name and game_type

    """
    request_url = str(API_ENDPOINT) + str('/team/')+str(team_id)

    response = requests.get(request_url)
    if response.status_code != 200:
        print("Fehler GET get_team")
        print(response.content)
        return "error", "error", "error"

    response_data = response.json()
    team_name = response_data['name']
    isis_name = response_data['isisName']
    game_type = response_data['type']

    print("name: "+str(team_name))
    print("isisName: "+str(isis_name))
    print("type: "+str(game_type))
    return team_name, isis_name, game_type


def create_player(player_name, team_token):
    """Create a new player for existing team.

    Returns
    player_id & player_token

    """
    data = {'name': player_name}
    token = "Basic " + str(team_token)
    auth = {'Authorization': token}
    request_url = str(API_ENDPOINT) + str('/players')

    response = requests.post(request_url, headers=auth, json=data)
    if response.status_code != 201:
        print("Fehler POST create_player")
        print(response.content)
        return "error", "error"

    response_data = response.json()
    player_id = response_data['id']
    player_token = response_data['token']

    print('player_id: ' + str(player_id))
    print('player_token: ' + str(player_token))
    return player_id, player_token


def test_player_token(player_token):
    """Test player_token.

    Returns
    player_id & valid boolean

    """
    valid = False
    token = "Basic " + str(player_token)
    auth = {'Authorization': token}
    request_url = str(API_ENDPOINT) + str('/playerlogin')

    response = requests.get(request_url, headers=auth)
    if response.status_code != 200:
        print("Fehler GET test_player_token")
        print(response.content)
        return "default", valid

    response_data = response.json()
    player_id = response_data['id']
    valid = response_data['valid']

    print('player_id: ' + str(player_id))
    print('valid: ' + str(valid))

    return player_id, valid


def list_players(count=100, start=0):
    """List all players.

    count and start limits the list received

    Returns
    list of players

    """
    players = {}
    request_url = (str(API_ENDPOINT) +
                   str('/players?count=') +
                   str(count)+str('&start=') +
                   str(start))

    response = requests.get(request_url)
    if response.status_code != 200:
        print("Fehler GET list_players")
        print(response.content)
        return players

    players = response.json()
    print("players: " + str(players))
    return players


def get_player(player_id):
    """Get player by player_id.

    Returns
    player_name, team_name

    """
    request_url = str(API_ENDPOINT) + str('/player/') + str(player_id)

    response = requests.get(request_url)
    if response.status_code != 200:
        print("Fehler GET get_player")
        print(response.content)
        return "error", "error"

    response_data = response.json()
    player_name = response_data['name']
    team_name = response_data['team']

    print("player_name: " + str(player_name))
    print("team_name: "+str(team_name))
    return player_name, team_name


def create_game(game_name, game_type, player_a, player_b,
                game_stats):
    """Create new game.

    Inputs
    game_name, game_type, player_token_a, player_token_b & game_stats
    game_stats include a string with board_state, time_budget & time_out

    Returns
    game_id

    """
    data = ({'name': game_name,
             'type': game_type,
             'players': {
                 'playerA': {
                     'id': player_a,
                     'timeout': game_stats['timeout'],
                     'initialTimeBudget': game_stats['timeBudget']
                 },
                 'playerB': {
                     'id': player_b,
                     'timeout': game_stats['timeout'],
                     'initialTimeBudget': game_stats['timeBudget']
                 }
             },
             'settings': {
                 'initialFEN': game_stats['initialFEN']
             }
             })
    request_url = str(API_ENDPOINT) + str('/games')

    response = requests.post(request_url, json=data)
    if response.status_code != 201:
        print("Fehler POST create_game")
        print(response.content)
        return "default"

    response_data = response.json()
    game_id = response_data['id']

    print('game_id: ' + str(game_id))
    return game_id


def list_games(state, count=100, start=0):
    """List all games.

    Inputs
    state, count & start
    state=[planned, running, completed]
    count and start limits the list received

    Returns
    list of games

    """
    games = {}
    request_url = (str(API_ENDPOINT) +
                   str('/games?count=') +
                   str(count)+str('&start=') +
                   str(start)+str('&state=') +
                   str(state))

    response = requests.get(request_url)
    if response.status_code != 200:
        print("Fehler GET list_games")
        print(response.content)
        return games

    games = response.json()

    print("games: " + str(games))
    return games


def get_game(game_id):
    """Get game by game_id.

    Returns
    game_name, game_type, players, settings, state
    players, settings and state are all json dictionaries

    """
    players = {"playerA": {"id": "default", "name": "default"},
               "playerB": {"id": "default", "name": "default"}}
    settings = {"initialFEN": "default", "timeBudget": 0, "timeout": 0}
    state = {"state": "default", "winner": "default", "fen": "default",
             "timeBudgets": {"playerA": 0, "playerB": 0}}
    request_url = str(API_ENDPOINT) + str('/game/') + str(game_id)

    response = requests.get(request_url)
    if response.status_code != 200:
        print("Fehler GET get_game")
        print(response.content)
        return "error", "error", players, settings, state

    response_data = response.json()
    game_name = response_data['name']
    game_type = response_data['type']
    players = info_players(response_data)
    settings = info_settings(response_data)
    state = info_state(response_data)
    # print("Players: " + str(players))
    # print("Settings: " + str(settings))
    # print("State: " + str(state))

    return game_name, game_type, players, settings, state


def send_event(game_id, player_token, action, move):
    """Send event to game-server.

    Inputs
    game_id, player_token, action & move
    action=['surrender', 'move']
    move is a uci string

    Returns
    valid boolean

    """
    data = {'type': action, 'details': {'move': move}}
    token = "Basic " + str(player_token)
    auth = {'Authorization': token}
    request_url = (str(API_ENDPOINT) + str('/game/') +
                   str(game_id) + str('/events'))

    response = requests.post(request_url, headers=auth, json=data)
    if response.status_code != 201:
        print("Fehler POST send_event")
        print(response.content)
        return "error"

    response_data = response.json()
    valid = response_data['valid']
    reason = response_data['reason']

    if valid is not True:
        print("No valid move: " + str(reason))

    return valid


def get_events(game_id, player_info, a0_game, network):
    """Get events and stays in loop until game ends.

    More to come

    """
    request_url = (str(API_ENDPOINT) + str('/game/') +
                   str(game_id) + str('/events'))

    response = requests.get(request_url, stream=True)
    if response.status_code != 200:
        print("Fehler GET get_events")
        print(response.content)
        return "error"

    for event in sseclient.SSEClient(response.url):
        if not event.data:
            continue
        print(event.data)
        data = json.loads(event.data)
        game_end, move_made, my_event = process_event(event.data, player_info)
        if game_end is True:
            winner = data["details"]["winner"]
            break
        if move_made is True and my_event is not True:
            move = data["details"]["move"]
            a0_game.make_move(move)
            color_next = get_color_fen(data)
            if color_next == player_info["player_color"]:
                valid = apply_move(game_id, player_info, a0_game, network)
                if valid is not True:
                    return "not valid"

    return winner


# ------------------------ Helper functions ----------------------------------#

def get_color_fen(data):
    """Get color for next players turn from fen."""
    fen = data["details"]["postFEN"]
    color = fen.split()[1]
    return color


def extract_player_info(player):
    """Extract player information from dictionary."""
    player_id = player["player_id"]
    player_token = player["player_token"]

    return player_id, player_token


def create_game_stats(board_state=START_POSITION, time_budget=TIME_BUDGET,
                      time_out=TIME_OUT):
    """Create game statistics.

    Inputs
    board_state, time_budget & time_out
    board_state -> default: START_POSITION
    time_budget -> default: TIME_BUDGET
    time_out -> default: TIME_OUT

    Returns
    game_stats as a string ready to feed create_game

    """
    game_stats = {'initialFEN': board_state,
                  'timeBudget': time_budget,
                  'timeout': time_out}

    return game_stats


def info_players(response_data):
    """Create players dictionary."""
    player_a_id = response_data['players']['playerA']['id']
    player_a_name = response_data['players']['playerA']['name']
    player_b_id = response_data['players']['playerB']['id']
    player_b_name = response_data['players']['playerB']['name']

    players = {"playerA": {"id": player_a_id, "name": player_a_name},
               "playerB": {"id": player_b_id, "name": player_b_name}}

    return players


def info_settings(response_data):
    """Create settings dictionary."""
    initial_fen = response_data['settings']['initialFEN']
    time_budget = response_data["players"]['playerA']['initialTimeBudget']
    time_out = response_data["players"]['playerA']['timeout']

    settings = {'initialFEN': initial_fen,
                'timeBudget': time_budget,
                'timeout': time_out}

    return settings


def info_state(response_data):
    """Create state dictionary."""
    state_g = response_data['state']['state']
    winner = response_data['state']['winner']
    fen = response_data['state']['fen']
    time_budget_a = response_data["players"]['playerA']['timeBudget']
    time_budget_b = response_data["players"]['playerB']['timeBudget']

    state = {"state": state_g,
             "winner": winner,
             "fen": fen,
             "timeBudgets": {"playerA": time_budget_a,
                             "playerB": time_budget_b}}

    return state


def player_mapping(player_id, players):
    """Map player_id to existing players in game.

    Returns
    player_color, player_nr
    player_color = ["w", "b"]
    player_nr = ["playerA", "playerB"]
    if player is playerA then player_color is white & vice versa

    """
    player_color = 'w'
    player_nr = 'playerA'
    if players['playerA']['id'] != player_id:
        player_color = 'b'
        player_nr = 'playerB'
    return player_color, player_nr


def player_time_budget(player_nr, state):
    """Get back time_budget of player."""
    time_budget_a = state['timeBudgets']['playerA']
    time_budget_b = state['timeBudgets']['playerB']

    if player_nr == 'playerA':
        return time_budget_a
    return time_budget_b


def process_event(data, player_info):
    """Process event to set game_end, move_made & my_event."""
    data = json.loads(data)
    event_type = data["type"]
    if event_type == "gameEnd":
        game_end = True
        move_made = False
        my_event = False
        return game_end, move_made, my_event

    if event_type == "move":
        game_end = False
        move_made = True
        if data["player"] == player_info["player_nr"]:
            my_event = True
        else:
            my_event = False
        return game_end, move_made, my_event

    game_end = False
    move_made = False
    my_event = False
    return game_end, move_made, my_event
