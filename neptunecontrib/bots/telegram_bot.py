#
# Copyright (c) 2019, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Spins of a Neptune bot with which you can interact on telegram

You can see which experiments are running, check the best experiements based
on defined metric and even plot it in Telegram.

Full list of options:
 * /project list NAMESPACE
 * /project select NAMESPACE/PROJECT_NAME
 * /project help
 * /experiments last NUMBER_OF_EXPERIMENTS
 * /experiments best METRIC_NAME NUMBER_OF_EXPERIMENTS
 * /experiments state STATE NUMBER_OF_EXPERIMENTS
 * /experiments help
 * /experiment link SHORT_ID
 * /experiment plot SHORT_ID METRIC_NAME OTHER_METRIC_NAME
 * /experiment help

Attributes:
    telegram_api_token(str): Your telegram bot api token.
        You can pass it either as --telegram_api_token or -t.
    neptune_api_token(str): Your neptune api token. If you
        set the NEPTUNE_API_TOKEN environemnt variable, you
        don't have to pass it here.
        You can pass it either as --neptune_api_token or -n.
        Default None.

Example:
    Spin off your bot::

        $ python neptunecontrib.bots.telegram
            --telegram_api_token 'a1249auscvas0vbia0fias0'
            --neptune_api_token 'asdjpsvdsg987das0f9sad0fjasdf='

    Go to your telegram and type.

    `/project list neptune-ml`

    Use help to see what is implemented.

     * '/project help'
     * '/experiments help'
     * '/experiemnt help'

"""

import argparse
from io import BytesIO

from neptune.sessions import Session
import matplotlib.pyplot as plt
import pandas as pd
from telegram.ext import Updater
from telegram.ext import CommandHandler, MessageHandler, Filters


class TelegramBot:
    def __init__(self, telegram_api_token, neptune_api_token):
        self.session = Session(api_token=neptune_api_token)
        self.updater = Updater(token=telegram_api_token)
        self.dispatcher = self.updater.dispatcher
        self.neptune_project = None
        self.project_name = None

        self.dispatcher.add_handler(CommandHandler('project', self.project, pass_args=True))
        self.dispatcher.add_handler(CommandHandler('experiments', self.experiments, pass_args=True))
        self.dispatcher.add_handler(CommandHandler('experiment', self.experiment, pass_args=True))
        self.dispatcher.add_handler(MessageHandler(Filters.command, self.unknown))

    def run(self):
        self.updater.start_polling()

    def project(self, bot, update, args):
        if args:
            if args[0] == 'select':
                self._project_select(bot, update, args)
            elif args[0] == 'list':
                self._project_list(bot, update, args)
            else:
                self._project_help(bot, update)
        else:
            self._project_help(bot, update)

    def experiments(self, bot, update, args):
        if not self.neptune_project:
            self._no_project_selected(bot, update)
        else:
            if args:
                if args[0] == 'last':
                    self._experiments_last(bot, update, args)
                elif args[0] == 'best':
                    self._experiments_best(bot, update, args)
                elif args[0] == 'state':
                    self._experiments_state(bot, update, args)
                else:
                    self._experiments_help(bot, update)
            else:
                self._experiments_help(bot, update)

    def experiment(self, bot, update, args):
        if not self.neptune_project:
            self._no_project_selected(bot, update)
        else:
            if args:
                if args[0] == 'link':
                    self._experiment_link(bot, update, args)
                elif args[0] == 'plot':
                    self._experiment_plot(bot, update, args)
                else:
                    self._experiment_help(bot, update)
            else:
                self._experiment_help(bot, update)

    def unknown(self, bot, update):
        bot.send_message(chat_id=update.message.chat_id,
                         text="Sorry, I only undestand /project, /experiments, /experiment")

    def _project_list(self, bot, update, args):
        if len(args) != 2:
            msg = ['message should have a format:',
                   '/project list NAMESPACE',
                   'for example:',
                   '/project list neptune-ml']
            msg = '\n'.join(msg)
        else:
            namespace = args[1]
            project_names = self.session.get_projects(namespace).keys()
            msg = '\n'.join(project_names)
        bot.send_message(chat_id=update.message.chat_id, text=msg)

    def _project_select(self, bot, update, args):
        if len(args) != 2:
            msg = ['message should have a format:',
                   '/project select NAMESPACE/PROJECT_NAME',
                   'for example:',
                   '/project select neptune-ml/neptune-examples']
            msg = '\n'.join(msg)
        else:
            self.project_name = args[1]
            namespace = self.project_name.split('/')[0]
            self.neptune_project = self.session.get_projects(namespace)[self.project_name]

            msg = 'Selected a project: {}'.format(self.project_name)
        bot.send_message(chat_id=update.message.chat_id, text=msg)

    def _project_help(self, bot, update):
        msg = """Available options are:\n
        /project list NAMESPACE
        /project select NAMESPACE/PROJECT_NAME
        """
        bot.send_message(chat_id=update.message.chat_id, text=msg)

    def _experiments_last(self, bot, update, args):
        if len(args) != 3:
            msg = ['message should have a format:',
                   '/experiments last NR_EXPS TIMESTAMP',
                   'for example:',
                   '/experiments last 5 finished']
            msg = '\n'.join(msg)
        else:
            nr_exps = int(args[1])
            timestamp = args[2]

            if timestamp not in ['created', 'finished']:
                msg = 'choose created or finished as timestamp'
            else:
                leaderboard = self.neptune_project.get_leaderboard()
                leaderboard[timestamp] = pd.to_datetime(leaderboard[timestamp])
                leaderboard.sort_values(timestamp, ascending=False, inplace=True)
                ids = leaderboard['id'].tolist()[:nr_exps]
                ids = ['id'] + ids
                msg = '\n'.join(ids)

        bot.send_message(chat_id=update.message.chat_id, text=msg)

    def _experiments_best(self, bot, update, args):
        if len(args) != 3:
            msg = ['message should have a format:',
                   '/experiments best METRIC NR_EXPS',
                   'for example:',
                   '/experiments best log_loss 3']
            msg = '\n'.join(msg)
        else:
            metric_name = args[1]
            metric_column = 'channel_' + metric_name
            nr_exps = int(args[2])

            leaderboard = self.neptune_project.get_leaderboard()
            scores = leaderboard.sort_values([metric_column], ascending=False)[['id', metric_column]]

            msg = 'id | {}\n'.format(metric_name)
            for idx, metric in scores.values[:nr_exps]:
                msg = msg + '{} | {}\n'.format(idx, metric)

        bot.send_message(chat_id=update.message.chat_id, text=msg)

    def _experiments_state(self, bot, update, args):
        if len(args) != 3:
            msg = ['message should have a format:',
                   '/experiments state STATE NR_EXPS',
                   'for example:',
                   '/experiments state running 4']
            msg = '\n'.join(msg)
        else:
            state = args[1]
            nr_exps = int(args[2])
            leaderboard = self.neptune_project.get_leaderboard(state=state)
            leaderboard['created'] = pd.to_datetime(leaderboard['created'])
            leaderboard.sort_values('created', ascending=False, inplace=True)
            ids = leaderboard['id'].tolist()[:nr_exps]
            ids = ['id'] + ids
            msg = '\n'.join(ids)
        bot.send_message(chat_id=update.message.chat_id, text=msg)

    def _experiments_help(self, bot, update):
        msg = """Available options are:\n
        /experiments last NR_EXPS
        /experiments best METRIC_NAME NR_EXPS(optional)
        /experiments state STATE NR_EXPS(optional)
        """
        bot.send_message(chat_id=update.message.chat_id, text=msg)

    def _experiment_link(self, bot, update, args):
        if len(args) != 2:
            msg = ['message should have a format:',
                   '/experiment link SHORT_ID',
                   'for example:',
                   '/experiment link NEP-508']
            msg = '\n'.join(msg)
        else:
            short_id = args[1]
            namespace, project = self.project_name.split('/')

            msg = 'https://app.neptune.ml/o/{}/org/{}/e/{}/details'.format(namespace, project, short_id)
        bot.send_message(chat_id=update.message.chat_id, text=msg)

    def _experiment_plot(self, bot, update, args):
        if len(args) < 3:
            msg = ['message should have a format:',
                   '/experiment plot SHORT_ID METRIC_NAME OTHER_METRIC_NAME',
                   'for example:',
                   '/experiment plot NEP-508 train_loss valid_loss']
            msg = '\n'.join(msg)
            bot.send_message(chat_id=update.message.chat_id, text=msg)
        else:
            short_id = args[1]
            if len(args) == 2:
                metric_names = [args[2]]
            else:
                metric_names = args[2:]

            experiment = self.neptune_project.get_experiments(id=short_id)[0]
            data = experiment.get_numeric_channels_values(*metric_names)

            fig = plt.figure()
            for channel_name in data.columns:
                if channel_name != 'x':
                    plt.plot('x', channel_name, data=data,
                             marker='', linewidth=2, label=channel_name)
            plt.legend()

            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            update.message.reply_photo(buffer)

    def _experiment_help(self, bot, update):
        msg = """Available options are:\n
        /experiment link SHORT_ID
        /experiments plot SHORT_ID METRIC_NAME (OTHER_METRIC_NAME) 
        """
        bot.send_message(chat_id=update.message.chat_id, text=msg)

    def _no_project_selected(self, bot, update):
        msg = ["You haven't selected your project.",
               "Do so by running:\n"
               "/project select NAMESPACE/PROJECT_NAME",
               "For example:",
               "/project select neptune-ml/neptune-examples"]
        msg = '\n'.join(msg)
        bot.send_message(chat_id=update.message.chat_id, text=msg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--telegram_api_token')
    parser.add_argument('-n', '--neptune_api_token', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()

    telegram_bot = TelegramBot(telegram_api_token=arguments.telegram_api_token,
                               neptune_api_token=arguments.neptune_api_token)

    while True:
        telegram_bot.run()
