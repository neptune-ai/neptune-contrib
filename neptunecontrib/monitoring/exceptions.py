#
# Copyright (c) 2021, Neptune Labs Sp. z o.o.
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

from neptune.exceptions import NeptuneException, STYLES


class NeptuneLegacyIncompatibilityException(NeptuneException):
    def __init__(self):
        message = """
{h1}
----NeptuneLegacyIncompatibilityException----------------------------------------
{end}
It seems you are passing a Run object, to a legacy integration which expects Experiment object.

What can I do?
    - Update your code to use the updated integration:
    https://docs.neptune.ai/integrations-and-supported-tools/intro
    - If you prefer to use the legacy integration, you can find their examples how to use theme here:
    https://docs-legacy.neptune.ai/integrations/index.html

{correct}Need help?{end}-> https://docs.neptune.ai/getting-started/getting-help
"""
        inputs = dict(list({}.items()) + list(STYLES.items()))
        super().__init__(message.format(**inputs))
