def reset_config(config):
    config['eur_held'] = 0
    config['usd_held'] = 100000
    config['commission'] = 0.0003
    config['sell_price'] = 1.5
    config['amount'] = 1
    config['lot_size'] = 100000
    config['value_per_pip'] = config['amount'] * config['lot_size'] / 10000
    config['buy_price'] = config['sell_price'] + config['commission']

# buy eur, there will be 3 cases here: price increase, decrease, and remain unchanged
def test_buy_action(config):
    print('-'*80)
    print('start test buy action')
    usd_held = config['usd_held']
    eur_held = config['eur_held']
    balance = usd_held
    # TEST CASE 1: buy 0.5 lot of EUR, price remain unchanged
    # start buy eur
    eur_held += config['amount'] * config['lot_size']
    usd_held -= config['amount'] * config['lot_size'] * config['buy_price']
    new_balance = eur_held * config['sell_price'] + usd_held
    print("new balance after buy: ", new_balance)
    assert new_balance == balance - round(config['value_per_pip'] * config['commission'] * 10000)

    # TEST CASE 2: price increase by 100 pip
    # 1 pip = 5 usd so we expect profit to go up by 500 usd
    config['sell_price'] += 0.010
    config['buy_price'] += 0.010
    balance = new_balance
    new_balance = usd_held + (eur_held * (config['sell_price']))
    print("new balance after price increase 100 pip: ", new_balance)
    assert new_balance == balance + (0.010 * 10000 * config['value_per_pip'])

    # TEST CASE 3: price decrease by 200 pip
    # 1 pip = 5 usd so we expect profit to go down by 1000 usd
    config['sell_price'] -= 0.020
    balance = new_balance
    new_balance = usd_held + (eur_held * (config['sell_price']))
    print("new balance after price decrease 200 pip: ", new_balance)
    assert new_balance == balance - (0.020 * 10000 * config['value_per_pip'])

    reset_config(config)
    print('end test buy action')

# sell eur, there will be 3 cases here: price increase, decrease, and remain unchanged
def test_sell_action(config):
    print('-' * 80)
    print('start test sell action')
    usd_held = config['usd_held']
    eur_held = config['eur_held']
    balance = usd_held

    # TEST CASE 1: sell 0.5 lot of EUR, price remain unchanged
    eur_held -= config['amount'] * config['lot_size']
    usd_held += config['amount'] * config['lot_size'] * config['sell_price']
    new_balance = eur_held * config['buy_price'] + usd_held
    print("new balance after sell: ", new_balance)
    assert new_balance == balance - round(config['value_per_pip'] * config['commission'] * 10000)

    # TEST CASE 2: price increase by 100 pip
    # 1 pip = 5 usd so we expect profit to go down by 500 usd
    config['buy_price'] += 0.010
    balance = new_balance
    new_balance = usd_held + (eur_held * (config['buy_price']))
    print("new balance after price increase 100 pip: ", new_balance)
    assert new_balance == balance - (0.010 * 10000 * config['value_per_pip'])

    # TEST CASE 3: price decrease by 200 pip
    # 1 pip = 5 usd so we expect profit to go up by 1000 usd
    config['buy_price'] -= 0.020
    balance = new_balance
    new_balance = usd_held + (eur_held * (config['buy_price']))
    print("new balance after price decrease 200 pip: ", new_balance)
    assert new_balance == balance + (0.020 * 10000 * config['value_per_pip'])

    reset_config(config)
    print('end test sell action')


# hold action, there will be 3 cases here: price increase, decrease, and remain unchanged
def test_hold_action(config):
    print('-' * 80)
    print('start test hold action')
    usd_held = config['usd_held']
    eur_held = config['eur_held'] + 100000
    balance = usd_held + eur_held * config['sell_price']

    # TEST CASE 1: balance after price increase by 100 pip and we hold positive eur amount
    config['sell_price'] += 0.01
    config['buy_price'] += 0.01
    new_balance = usd_held + eur_held * config['sell_price']
    print("new balance after holding price increase by 100 pip: ", new_balance)
    assert new_balance == balance + (0.010 * 10000 * config['value_per_pip'])

    # TEST CASE 2: balance after price decrease by 300 pip and we hold positive eur amount
    balance = new_balance
    config['sell_price'] -= 0.03
    config['buy_price'] -= 0.03
    new_balance = usd_held + eur_held * config['sell_price']
    print("new balance after holding price decrease by 300 pip: ", new_balance)
    assert new_balance == balance - (0.030 * 10000 * config['value_per_pip'])

    print('end test hold action')


def run_test(config):
    test_buy_action(config)
    test_sell_action(config)
    test_hold_action(config)
    print('-' * 80)
    print("all test passed")


if __name__ == '__main__':
    config = {'eur_held': 0,
              'usd_held': 100000,
              'commission': 0.0003,
              'sell_price': 1.5,
              'amount': 1,
              'lot_size': 100000}
    config['value_per_pip'] = config['amount'] * config['lot_size'] / 10000
    config['buy_price']= config['sell_price'] + config['commission']
    run_test(config)