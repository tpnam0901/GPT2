# Changelog

## [0.2.0](https://github.com/tpnam0901/PrivateGPT/compare/v0.1.0...v0.2.0) (2024-08-30)


### Features

* add change vocab size for model ([bf1ff05](https://github.com/tpnam0901/PrivateGPT/commit/bf1ff05ff2dd69a690f01be9e188d0f2c301cc4c))
* add custom dataset preparation ([580e809](https://github.com/tpnam0901/PrivateGPT/commit/580e809f5134b51dad14127fc17fc1a63a4da415))
* add hosting server ([78f5ffb](https://github.com/tpnam0901/PrivateGPT/commit/78f5ffb9105a853a7f7a2b57be44c35a7667003b))
* add inference script ([a5f177d](https://github.com/tpnam0901/PrivateGPT/commit/a5f177d5de7391a1f8604d6661d8d9b23bf0ccbc))
* **server_api:** add optional checkpoint ([4099e95](https://github.com/tpnam0901/PrivateGPT/commit/4099e958db78999feec0fc4f95097314131163a6))


### Bug Fixes

* duplicate cfg.log file ([960f530](https://github.com/tpnam0901/PrivateGPT/commit/960f5300a6a121d72ebac9ba8444d43e8aabfc4b))
* duplicate log in mlflow ([bd83877](https://github.com/tpnam0901/PrivateGPT/commit/bd838770e01f5f411ff167992d3039cf0483ee08))
* eos_token, response message, wrong key json ([f75d810](https://github.com/tpnam0901/PrivateGPT/commit/f75d8102e08652e0780110076e9ebcb003c026d0))
* infer token of server api, add eot token to input ([20d13fe](https://github.com/tpnam0901/PrivateGPT/commit/20d13fec708c0c4140f8c86a16067afc4e24346b))
* miss best ckpt, load config ([7616f7c](https://github.com/tpnam0901/PrivateGPT/commit/7616f7c253bc849954ca4e73eeb0c404a331bf43))
* not passing parameters in the config ([59ed96a](https://github.com/tpnam0901/PrivateGPT/commit/59ed96a790c869973f26dee4eff1fb66587eb9b2))
* **server_api:** bot repeat the same sentence ([4f7c717](https://github.com/tpnam0901/PrivateGPT/commit/4f7c717cd5e0646c56b3825c5dc33355c8c02bc6))
* **server_api:** special token ([02f9e4d](https://github.com/tpnam0901/PrivateGPT/commit/02f9e4dafade241c6bc838ba633b1328a5172c90))
* variable type in args ([c164c7e](https://github.com/tpnam0901/PrivateGPT/commit/c164c7e749e554134b8d2d091d2081c5aa142681))


### Documentation

* add export weight ([0438c89](https://github.com/tpnam0901/PrivateGPT/commit/0438c890554e7a66e5ccfc67e9c571e165038bf8))
* fix username, add inference scripts ([17ed15c](https://github.com/tpnam0901/PrivateGPT/commit/17ed15c17f63468446e54965890fed02c9ca0495))

## 0.1.0 (2024-08-28)


### Features

* add automatic mix precision ([5c4831d](https://github.com/tpnam0901/PrivateGPT/commit/5c4831d90b52232ca73457f031fa299a583361dc))
* add crop block size of GPT model ([3c66a57](https://github.com/tpnam0901/PrivateGPT/commit/3c66a57ca416f3574661819959b5533c78da1374))
* add data distributed parallel training ([f0bb8b0](https://github.com/tpnam0901/PrivateGPT/commit/f0bb8b0f778297e5e7f0cba3be9bf68977325dba))
* add dataloader for .bin file - openwebtext dataset ([2966136](https://github.com/tpnam0901/PrivateGPT/commit/29661368b8ee297d578e710c8a279666802d691c))
* add GPT2 config, weight converter, missing model file ([c2ee7d7](https://github.com/tpnam0901/PrivateGPT/commit/c2ee7d71229479cfa6c417bb6a87c16098f752ae))
* add GPT2 model ([bb53955](https://github.com/tpnam0901/PrivateGPT/commit/bb53955f9fe24564aa7c97371c89910e1244a5e4))
* add log interval ([6640c38](https://github.com/tpnam0901/PrivateGPT/commit/6640c3893e2c52fbad7ac310cb31c45026ae65ec))
* add loss function, learning rate schedular ([4a5f625](https://github.com/tpnam0901/PrivateGPT/commit/4a5f6253b76a267c047a846364edbdb050beba6d))
* add openwebtext preprocess ([5d7daa9](https://github.com/tpnam0901/PrivateGPT/commit/5d7daa9834b496d175d0658c735e51c72993588d))
* add optimizers ([634108f](https://github.com/tpnam0901/PrivateGPT/commit/634108febd9255e4f25e9ee067b649912f73bb47))
* add train config for openwebtext dataset, gpt2 model ([92d20bd](https://github.com/tpnam0901/PrivateGPT/commit/92d20bdc4ce2ad1fe24f8871310d18f904671f6b))
* add train script for GPT2, update config ([13a3fb3](https://github.com/tpnam0901/PrivateGPT/commit/13a3fb37d25fab95ce01180a283440acf434f04a))


### Bug Fixes

* automatic mix precision for cpu ([ffe3058](https://github.com/tpnam0901/PrivateGPT/commit/ffe30584f0401afe01b42ffa2565b371b31b03ee))
* change cel to functional ([66c25ef](https://github.com/tpnam0901/PrivateGPT/commit/66c25efb39f4e49c4343613d8ec5c3cfb6bc6d54))
* duplicate log ([a903e81](https://github.com/tpnam0901/PrivateGPT/commit/a903e81245db641af25b3faeba5041d9959b9d31))
* feed forward loss function, change variable name ([6d799dd](https://github.com/tpnam0901/PrivateGPT/commit/6d799dd2cc75fc448a1262afd8d6104abf22a869))
* **gpt.py:** remove model ff return loss ([f12aa99](https://github.com/tpnam0901/PrivateGPT/commit/f12aa99c533ee06ca2c4d29af362863c8b76e47f))
* infinity validating step ([a07d7b6](https://github.com/tpnam0901/PrivateGPT/commit/a07d7b6211de1976075fbc4e27d9e5ef47465df9))
* **losses.py:** missing parent init function ([2b6a2df](https://github.com/tpnam0901/PrivateGPT/commit/2b6a2df5780f84b186f5b677ebf980771d8bfb97))
* openwebtext dataset link ([4fe8add](https://github.com/tpnam0901/PrivateGPT/commit/4fe8add8af61b93619203c77d6c45dbb6c14a3df))
* same GPU in training multi GPU ([b5ddc45](https://github.com/tpnam0901/PrivateGPT/commit/b5ddc450b68e2118bebe81c4692dd5cad846c702))
* step calling in optimizer ([4f1d309](https://github.com/tpnam0901/PrivateGPT/commit/4f1d3094c562c7939cce33d4b0bc8571913688c6))
* x, y device in dataloader ([9db7aad](https://github.com/tpnam0901/PrivateGPT/commit/9db7aad00cd6b0fef5e05461cea5f9e1d6b44d79))


### Documentation

* add training instruction ([bcfc7b1](https://github.com/tpnam0901/PrivateGPT/commit/bcfc7b11e4f0c5a191ee86a3160790356a7e36ce))
