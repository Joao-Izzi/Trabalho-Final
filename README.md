# Esse projeto é o trabalho final proposto para a disciplina de Aprendizado de Maquina do Mestrado em Estatística do Pipges. O seguinte problema foi proposto:

# Classificação de Bons e Maus Pagadores

Um cliente do setor financeiro forneceu um conjunto de dados contendo informações de diferentes clientes, com o objetivo de identificar bons e maus pagadores. Cada registro do dataset representa um cliente, descrito por diversas variáveis que indicam características financeiras e comportamentais, além da indicação se o cliente é um bom pagador (classe 0) ou mau pagador (classe 1).

O cliente precisa de uma solução automatizada para classificar novos clientes em bons ou maus pagadores de forma eficaz, para otimizar processos de concessão de crédito e minimizar riscos financeiros.

Por se tratar de decisões que envolvem dinheiro real e impacto direto nas operações da empresa, a solução deve atender aos seguintes requisitos essenciais:

- **Desempenho preditivo confiável:** o modelo deve apresentar bom desempenho para garantir decisões adequadas em dados futuros.
- **Interpretabilidade:** o cliente exige que a solução seja compreensível, de modo que os analistas possam identificar quais características influenciam as decisões e justificar os resultados internamente e para órgãos reguladores.
- **Automação:** o processo deve ser automatizado, abrangendo desde o pré-processamento dos dados até a seleção das variáveis mais relevantes e a construção do modelo final.

Sua tarefa é desenvolver uma solução automatizada que atenda a esses objetivos.

A solução entregue deve conter código completo, organizado e documentado, facilitando sua integração ao fluxo operacional do cliente.

- **O código entregável é:** Entregavel_pipe_credit
- O codigo chamado **Analise.py** foi um codigo usado para uma analise inicial e alguns testes

No entregável eu entrego apenas o melhor modelo, a análise descritiva e tunning de hiperparâmetros usado para chegar nele e por fim como é dada a predição de novos clientes e a interpretração feita.

# **Comentários finais e relevantes sobre meu raciocínio**
- Eu só acredito que esses resultados extremamente absurdos e bons que obtive no KNN estão certos, por que eu sei que os dados foram gerados sinteticamente pelo meu professor e que existia uma resposta correta.
- Tenho total consciência de que não é pra existir um modelo bom daquele, to confiando que eu cheguei na resposta que era pra chegar kkkkk
- A variável 31 é um vazamento direto das informações dos dados, por isso deve ser excluída.
- Cada uma das três features utilizadas como covariáveis sozinhas não são um vazamento direto da resposta, e por isso, acredito que o resultado final que cheguei é valido.
- Além disso eu fiz um teste rápido retirando as feat 8, 17, 31 e 50 e rodei um RF  e um boost e o resultado deles foram um modelo aleatório, então isso me leva a acreditar mais ainda que utilizar a combinação das feat 8, 17 e 50 é o que leva o modelo a acertar